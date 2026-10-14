from typing import List, Any

import onnx

from ..diagnostics import issue


# ONNX 元素类型 → (元素字节数, 精度名) 单一事实来源。
#   精度名与 hardware.yaml 的 peak_flops 键对齐 (未收录精度时下游回退默认精度);
#   字节数用于访存量估算。此前字节表与精度名分列两处, 增删类型须同改两张表,
#   易漏改漂移, 故合并为一张。变长类型 (STRING) 无固定字节数, 记为 None,
#   由 get_dtype_info 兜底。
ONNX_TYPE_INFO = {
    onnx.TensorProto.FLOAT: (4, "fp32"),
    onnx.TensorProto.UINT8: (1, "uint8"),
    onnx.TensorProto.INT8: (1, "int8"),
    onnx.TensorProto.UINT16: (2, "uint16"),
    onnx.TensorProto.INT16: (2, "int16"),
    onnx.TensorProto.INT32: (4, "int32"),
    onnx.TensorProto.INT64: (8, "int64"),
    onnx.TensorProto.STRING: (None, "string"),
    onnx.TensorProto.BOOL: (1, "bool"),
    onnx.TensorProto.FLOAT16: (2, "fp16"),
    onnx.TensorProto.DOUBLE: (8, "fp64"),
    onnx.TensorProto.UINT32: (4, "uint32"),
    onnx.TensorProto.UINT64: (8, "uint64"),
    onnx.TensorProto.COMPLEX64: (8, "c64"),
    onnx.TensorProto.COMPLEX128: (16, "c128"),
    onnx.TensorProto.BFLOAT16: (2, "bf16"),
    onnx.TensorProto.FLOAT8E4M3FN: (1, "fp8"),
    onnx.TensorProto.FLOAT8E4M3FNUZ: (1, "fp8"),
    onnx.TensorProto.FLOAT8E5M2: (1, "fp8"),
    onnx.TensorProto.FLOAT8E5M2FNUZ: (1, "fp8"),
}

# 未收录类型 / 变长类型 (如 STRING) 的兜底元素字节数 (取 fp32 宽度),
# 避免下游 numel * None 崩溃; 构建期另发 missing_dtype_size 告警提示。
DEFAULT_DTYPE_SIZE = 4

SCALAR_TENSOR_SHAPE = [1]


def get_dtype_info(elem_type):
    """ONNX 元素类型 → (字节数, 精度名)。

    未收录类型: 字节数回退 :data:`DEFAULT_DTYPE_SIZE`, 精度名回退该元素的
    ONNX 枚举名 (小写, 如 ``int4``; 非合法枚举值时回退 None, 下游按设备默认
    精度计); 变长类型 (STRING) 字节数同样回退默认值。
    如需判断是否为"确定的定宽类型", 见 :func:`has_dtype_size`。
    """
    size, name = ONNX_TYPE_INFO.get(elem_type, (DEFAULT_DTYPE_SIZE, None))
    if name is None:
        # 未收录类型: 尽量给出可读精度名, 便于诊断与后续精度表扩展。
        try:
            name = onnx.TensorProto.DataType.Name(elem_type).lower()
        except (ValueError, TypeError):
            name = None
    return (size or DEFAULT_DTYPE_SIZE, name)


def has_dtype_size(elem_type) -> bool:
    """该 ONNX 类型是否已有确定的元素字节数 (未收录或变长类型返回 False)。"""
    info = ONNX_TYPE_INFO.get(elem_type)
    return bool(info and info[0])


def numel(shape: List[Any]) -> int:
    """
    计算张量的元素数量（各维相乘）。

    要求 shape 的每一维均为正整数: 动态维度已在入口 validate_static_shapes
    处被拒绝, 故此处不再处理动态维度或批量大小。空形状与非整数维度
    (None / dim_param 字符串) 视为不可计, 返回 0。
    """
    if not shape:  # 空形状
        return 0

    total = 1
    for dim in shape:
        if not isinstance(dim, int):
            return 0
        total *= dim
    return total


def safe_axis(axis, rank):
    """把 *axis* 规范化为 ``[0, rank)``; rank 非正 (标量) 时返回 None。"""
    if axis is None:
        return None
    if rank <= 0:  # 标量无轴可归一, 避免 % 0
        return None

    return (axis + rank) % rank


def get_default_attrs(op_type: str, version: int = None):
    """取算子 schema 的默认属性表 (未注册 schema 时降级为空表)。

    未注册 schema (com.microsoft 域算子 / 自定义算子, 如 Attention) 无法取默认值,
    此处降级为 unknown_op 告警并返回 ``{}``, 交由各 handler 的手写默认值兜底,
    而不是让 :class:`onnx.defs.SchemaError` 中断整条流水线。
    """
    try:
        # 获取算子 schema（若不指定 version，默认为最新）
        schema = (
            onnx.defs.get_schema(op_type, version)
            if version
            else onnx.defs.get_schema(op_type)
        )
    except onnx.defs.SchemaError:
        issue(
            "unknown_op",
            f"No schema registered for '{op_type}', 无默认属性可注入, 由算子自身属性兜底",
            op_type=op_type,
        )
        return {}

    defaults = {}
    for name, attr in schema.attributes.items():
        # AttributeProto.default_value 恒为 proto 对象 (never Python None);
        # type == UNDEFINED 才表示该属性无真实默认值 (通常为可选属性)。
        # 按“有值则替换”语义: 无真实默认值时跳过、不注入 key, 避免 op_args 出现
        # 多余的 null (op_args 会序列化展示到网页), 交由各 handler 的手写默认值兜底。
        if attr.default_value.type == onnx.AttributeProto.UNDEFINED:
            continue
        defaults[name] = onnx.helper.get_attribute_value(attr.default_value)
    return defaults


def decode_bytes(obj):
    """递归解码 bytes 为 str，支持 dict/list/tuple 嵌套结构"""
    if isinstance(obj, bytes):
        return obj.decode("utf-8")
    elif isinstance(obj, list):
        return [decode_bytes(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(decode_bytes(x) for x in obj)
    elif isinstance(obj, dict):
        return {decode_bytes(k): decode_bytes(v) for k, v in obj.items()}
    else:
        return obj
