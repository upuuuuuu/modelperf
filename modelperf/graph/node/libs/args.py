import onnx

from ...utils import get_default_attrs, decode_bytes
from ....diagnostics import issue


def get_arg(
    args, key, default=None, required=False, treat_none_as_missing=True, op_type=None
):
    """从算子属性字典 *args* 中取参数 *key* 的唯一实现。

    参数
    ----
    args : dict
        算子属性字典（如 ``node.args``）。
    key : str
        属性名。
    default : Any
        缺失时使用的默认值。
    required : bool
        为 True 时缺失且 ``default is None`` 则抛 ``KeyError``；若给了 default
        则告警并返回 default。
    treat_none_as_missing : bool
        为 True 时显式 ``None`` 也视为缺失（统一 computation / memory 两侧语义）。
    op_type : str | None
        算子类型，仅用于错误/告警信息。

    返回
    ----
    Any
        属性值或 *default*。
    """
    if not isinstance(args, dict):
        args = {}
    value = args.get(key)
    missing = (key not in args) or (treat_none_as_missing and value is None)
    if not missing:
        return value

    if required:
        if default is None:
            issue(
                "missing_required_arg",
                f"Missing required operator argument: '{key}' in {op_type}",
                op_type=op_type,
                arg=key,
            )
        issue(
            "missing_arg",
            f"Using default value {default} for argument '{key}' in {op_type}",
            op_type=op_type,
            arg=key,
        )
        return default
    return default


def get_node_attrs_dict(node: onnx.NodeProto):
    """
    返回 ONNX node 的完整属性字典，包括默认值和实际属性

    参数:
        node: onnx.NodeProto
        op_type: 算子类型，如 "ReduceMean"
        opset_version: ONNX opset 版本

    返回:
        dict: {attr_name: attr_value}
    """
    # 获取算子默认属性
    attrs_dict = get_default_attrs(node.op_type)
    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.TENSOR:
            tensor = onnx.helper.get_attribute_value(attr)
            # 优先返回 dims，如果没有 dims 再返回实际数据
            if tensor.dims:
                attr_value = list(tensor.dims)
            elif tensor.float_data:
                attr_value = list(tensor.float_data)
            elif tensor.int32_data:
                attr_value = list(tensor.int32_data)
            elif tensor.int64_data:
                attr_value = list(tensor.int64_data)
            elif tensor.string_data:
                attr_value = [s.decode("utf-8") for s in tensor.string_data]
            else:
                attr_value = None
        else:
            attr_value = onnx.helper.get_attribute_value(attr)
        attrs_dict[attr.name] = attr_value

    return decode_bytes(attrs_dict)
