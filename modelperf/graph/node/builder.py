from .definition import Node
from .libs import get_node_attrs_dict, Computation, MemoryAccess


def make_node_ir(onnx_node, node_id, in_tensors, out_tensors, p_tensors):
    node = Node()
    # 用上层算出的唯一 id (而非 onnx_node.name), 保证 feature["id"] 与图 key 一致;
    # ONNX 节点名可能为空或重复, 直接取用会导致 CG 节点互相覆盖。
    node.id = node_id
    node.type = onnx_node.op_type
    node.args = get_node_attrs_dict(onnx_node)

    node.in_tensor_infos = in_tensors
    node.out_tensor_infos = out_tensors
    node.p_tensor_infos = p_tensors

    return node


def make_node_cg(node_ir):
    node = Node.from_dict(node_ir)
    op_type = node.type
    args = node.args

    in_tensors = node.in_tensor_infos
    out_tensors = node.out_tensor_infos
    p_tensors = node.p_tensor_infos

    # find and set computation and memory info
    node.computation_info = Computation(
        op_type, args, in_tensors, out_tensors, p_tensors
    ).build()
    node.memory_info = MemoryAccess(
        op_type, args, in_tensors, out_tensors, p_tensors
    ).build()

    return node
