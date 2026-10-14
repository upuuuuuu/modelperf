import onnx
import networkx as nx

from .node.operators import LEARNABLE_OPS
from .node import make_node_ir, make_node_cg
from .edge import make_edge_ir, make_edge_cg

from ..diagnostics import issue
from .utils import (
    DEFAULT_DTYPE_SIZE,
    get_dtype_info,
    has_dtype_size,
    SCALAR_TENSOR_SHAPE,
)


def parse_shape(ttype):
    if not ttype.HasField("shape"):
        return None
    dims = []
    for dim in ttype.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append(None)
    return dims if dims else SCALAR_TENSOR_SHAPE


def build_tensor_map(graph):
    missing_dtype_sizes = []

    def make_info(name, shape, elem_type):
        dtype_size, dtype = get_dtype_info(elem_type)
        if not has_dtype_size(elem_type):
            missing_dtype_sizes.append(elem_type)
        return {
            "id": name,
            "shape": shape,
            "dtype_size": dtype_size,
            "dtype": dtype,
        }

    # initializer map
    initializer_map = {
        init.name: make_info(
            init.name,
            list(init.dims) if init.dims else SCALAR_TENSOR_SHAPE,
            init.data_type,
        )
        for init in graph.initializer
    }
    tensor_map = dict(initializer_map)

    # other tensors
    sources = [graph.input, graph.value_info, graph.output]
    for src in sources:
        for item in src:
            if item.type.HasField("tensor_type"):
                ttype = item.type.tensor_type
                tensor_map[item.name] = make_info(
                    item.name, parse_shape(ttype), ttype.elem_type
                )

    # dtype 大小缺失告警 (未收录/变长类型已按默认宽度兜底)
    if missing_dtype_sizes:
        unique_missing = sorted(set(missing_dtype_sizes))
        issue(
            "missing_dtype_size",
            f"{len(unique_missing)} tensor dtype(s) have no fixed size, "
            f"e.g. {unique_missing[:5]}; 已按 {DEFAULT_DTYPE_SIZE} 字节估算",
            count=len(unique_missing),
        )

    return tensor_map, initializer_map


def get_tensor_info(tensor_map, tensor_name):
    info = tensor_map.get(tensor_name)
    if info is None:
        return {"id": tensor_name, "shape": None, "dtype_size": None, "dtype": None}
    return info


def _make_node_names_unique(graph):
    """为每个 ONNX 节点计算唯一 id (不改写 node.name)。

    返回与 ``graph.node`` 等长的 id 列表, 由调用方按拓扑序取用。无名节点用
    ``__{op_type}_{i}``, 重名节点在原名字后追加 ``__{i}`` 予以区分。

    torch 导出与图简化会产生无名或同名的算子 (如 GNN 里的 MatMul/Split),
    而 CG 以节点名为 id, 重名会导致节点互相覆盖、边找不到 tensor 信息。
    此处只产出 id 而不就地改写 ``node.name``, 避免对入参模型产生副作用。
    """
    used = set()
    node_ids = []
    for i, node in enumerate(graph.node):
        name = node.name
        if not name:
            name = f"__{node.op_type}_{i}"
        elif name in used:
            name = f"{name}__{i}"
        while name in used:
            name += "_"
        used.add(name)
        node_ids.append(name)
    return node_ids


def build_ir(model: onnx.ModelProto, batch_size: int) -> nx.DiGraph:
    """
    从 ONNX 模型构建 IR（networkx DiGraph）。
    """
    graph = model.graph
    node_ids = _make_node_names_unique(graph)
    tensor_map, initializer_map = build_tensor_map(graph)

    cg = nx.DiGraph()
    # 仅作图级元数据 (可视化展示用), 不参与 shape / flops / mac / 代价计算
    cg.graph["batch_size"] = batch_size

    tensor_producer = {}
    missing_shapes = []
    # ONNX 规范要求 graph.node 拓扑有序, 故此处直接顺序遍历
    for node, node_id in zip(graph.node, node_ids):
        # 输入张量收集
        in_tensors, p_tensors = [], []
        for input_name in node.input:
            if not input_name:
                continue

            tensor_info = get_tensor_info(tensor_map, input_name)
            if tensor_info["shape"] is None:
                missing_shapes.append(input_name)

            if input_name in initializer_map and node.op_type in LEARNABLE_OPS:
                p_tensors.append(tensor_info)
            else:
                in_tensors.append(tensor_info)

        # 输出张量收集
        out_tensors = []
        for output_name in node.output:
            if not output_name:
                continue

            tensor_info = get_tensor_info(tensor_map, output_name)
            if tensor_info["shape"] is None:
                missing_shapes.append(output_name)

            out_tensors.append(tensor_info)
            tensor_producer[output_name] = node_id

        # 构建节点 feature
        node_feature = make_node_ir(
            node,
            node_id,
            in_tensors=in_tensors,
            out_tensors=out_tensors,
            p_tensors=p_tensors,
        )
        cg.add_node(node_id, feature=node_feature.to_dict_ir())

        # 添加边
        for input_name in node.input:
            if input_name and input_name in tensor_producer:
                src = tensor_producer[input_name]
                tgt = node_id
                edge_feature = make_edge_ir(src_id=src, tgt_id=tgt)
                cg.add_edge(src, tgt, feature=edge_feature.to_dict_ir())

    # missing shapes 告警 (可降级: 下游按缺省值处理)
    if missing_shapes:
        unique_missing = sorted(set(missing_shapes))
        issue(
            "missing_shape",
            f"{len(unique_missing)} tensors have no shape info, "
            f"e.g. {unique_missing[:5]}",
            count=len(unique_missing),
        )

    return cg


def build_cg(ir: nx.DiGraph) -> nx.DiGraph:
    """
    从 IR 图构建计算图（CG）。
    """
    cg = nx.DiGraph()
    cg.graph.update(ir.graph)

    # build nodes
    for node_id, data in ir.nodes(data=True):
        node_feature_ir = data["feature"]
        node_feature_cg = make_node_cg(node_feature_ir)
        cg.add_node(node_id, feature=node_feature_cg.to_dict_cg())

    # build edges (depends on CG node features)
    for src, tgt, data in ir.edges(data=True):
        edge_feature_ir = data["feature"]
        edge_feature_cg = make_edge_cg(cg, edge_feature_ir)  # in: cg not ir
        cg.add_edge(src, tgt, feature=edge_feature_cg.to_dict_cg())

    return cg
