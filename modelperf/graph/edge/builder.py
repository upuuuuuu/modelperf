from .definition import Edge, TensorInfo

from ..utils import numel
from ...diagnostics import issue


def make_edge_ir(src_id, tgt_id):
    edge = Edge()
    edge.id = f"{src_id}->{tgt_id}"
    edge.src = src_id
    edge.tgt = tgt_id

    return edge


def make_edge_cg(cg, edge_ir):
    edge = Edge().from_dict(edge_ir)
    src_id = edge.src
    tgt_id = edge.tgt

    # find delivered tensor info
    tgt_in_tensor_infos = cg.nodes[tgt_id]["feature"]["in_tensor_infos"]
    src_out_tensor_infos = cg.nodes[src_id]["feature"]["out_tensor_infos"]
    delivered_tensor_info = None
    for in_tensor_info in tgt_in_tensor_infos:
        in_id = in_tensor_info["id"]
        for out_tensor_info in src_out_tensor_infos:
            if out_tensor_info["id"] == in_id:
                delivered_tensor_info = in_tensor_info
                break
        if delivered_tensor_info is not None:
            break
    if delivered_tensor_info is None:
        # 合法 ONNX 图不可达: 仅当同一张量既是 initializer 又是节点输出
        # (张量重名, onnx.checker 即拒绝) 时才出现, 属用户模型非法。
        issue(
            "invalid_input",
            f"图结构非法: 边 {edge.id} ({src_id} -> {tgt_id}) 的传递张量无法定位; "
            f"请检查模型是否存在重复定义的张量名",
        )

    # set delivered tensor info
    tensor_info = TensorInfo()
    shape = delivered_tensor_info["shape"]
    dtype_size = delivered_tensor_info["dtype_size"]
    tensor_info.tensor_shape = shape
    tensor_info.tensor_dtype = delivered_tensor_info.get("dtype", "")
    tensor_info.tensor_dtype_size = dtype_size
    tensor_info.tensor_bytes = (numel(shape) or 0) * (dtype_size or 0)
    edge.delivered_tensor_info = tensor_info

    return edge
