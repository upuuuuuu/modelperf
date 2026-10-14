"""入口输入校验: 模型 / 数据 / 硬件设备, 以及 ONNX 图的动态维度拒绝。

ModelPerf 的代价计算依赖完全静态的形状 (numel 要求每一维都是正整数),
因此入口处统一做校验并尽早抛出清晰错误, 而不是等到下游计算时才崩溃。

所有校验统一走 :func:`diagnostics.issue`, 由策略表 (默认 ERROR) 决定抛
:class:`InvalidInputError` / :class:`DynamicShapeError`; 调用方可用
``configure(invalid_input="warn")`` 等临时放宽。
"""

from __future__ import annotations

import os
from typing import Any, List

from .diagnostics import issue


# any2onnx 支持的模型格式, 外加直接读取的 seer_ir
#   - torch 单 key: nn.Module / TorchScript 同走 torch.onnx.export
#   - tensorflow 族五 key: keras / tensorflow(=Keras) / tf_savedmodel / tf_graphdef / tflite
SUPPORTED_MODEL_FORMATS = (
    "seer_ir",
    "onnx",
    "torch",
    "tensorflow",
    "keras",
    "tf_savedmodel",
    "tf_graphdef",
    "tflite",
    "chainer",
    "mxnet",
    "paddle",
    "caffe",
    "jax",
    "mindspore",
    "oneflow",
)

# 必须提供 data_dict、且以“形状字典”规范表达的格式 (由 ModelPerf 内部构造张量)。
# 其余格式 (onnx / tf_savedmodel / tf_graphdef / tflite / paddle / caffe / seer_ir)
# 自带签名或输入节点, 无需 data_dict。
DATA_DICT_REQUIRED_FORMATS = (
    "torch",
    "tensorflow",
    "keras",
    "chainer",
    "mxnet",
    "jax",
    "mindspore",
    "oneflow",
)

# 必须提供已存在路径 (文件或目录) 作为 model 的格式
PATH_REQUIRED_FORMATS = (
    "seer_ir",
    "tf_savedmodel",
    "tf_graphdef",
    "tflite",
    "paddle",
    "caffe",
)

# 形状字典允许的 dtype 白名单 (单一事实来源, 适配器据此构造张量)
# 命名与 torch/tf 的 dtype 属性一致; bfloat16 无 numpy 等价类型
SUPPORTED_DTYPES = (
    "float32",
    "float16",
    "bfloat16",
    "float64",
    "int8",
    "uint8",
    "int16",
    "int32",
    "int64",
    "bool",
)


def _validate_input_spec(data_dict: Any) -> None:
    """强制 data_dict 为形状字典: {输入名: {"shape": (正整数, ...), "dtype": <白名单>}}。

    key 需为模型 forward 形参名; shape 全静态且每维为正整数; dtype 取自白名单。
    ModelPerf 会据此构造示例张量, 因此不接受真张量 / TensorSpec 等形态。
    """
    if not isinstance(data_dict, dict) or not data_dict:
        issue(
            "invalid_input",
            "data_dict 需为非空字典 {输入名: {'shape': (...), 'dtype': '...'}}; "
            f"收到: {data_dict!r}",
        )

    for name, spec in data_dict.items():
        if not isinstance(spec, dict) or "shape" not in spec or "dtype" not in spec:
            issue(
                "invalid_input",
                f"data_dict[{name!r}] 需为 {{'shape': (...), 'dtype': '...'}}; "
                f"收到: {spec!r}",
            )

        shape = spec["shape"]
        if (
            isinstance(shape, (str, bytes))
            or not isinstance(shape, (list, tuple))
            or len(shape) == 0
        ):
            issue(
                "invalid_input",
                f"data_dict[{name!r}]['shape'] 需为非空形状元组; 收到: {shape!r}",
            )
        for dim in shape:
            if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
                issue(
                    "invalid_input",
                    f"data_dict[{name!r}]['shape'] 每维需为正整数; 收到: {shape!r}",
                )

        if spec["dtype"] not in SUPPORTED_DTYPES:
            issue(
                "invalid_input",
                f"data_dict[{name!r}]['dtype'] 需为 {list(SUPPORTED_DTYPES)} 之一; "
                f"收到: {spec['dtype']!r}",
            )


def validate_build_inputs(
    model: Any,
    model_format: str,
    data_dict: Any,
    batch_size: int,
) -> None:
    """校验 ``build_computation_graph`` 的入口参数 (非法即抛, 纯 fail-fast)。

    可选的 devices 不在此校验: 未知设备属可降级情形, 由 inject_cost 自行告警并跳过。
    """
    # model
    if not isinstance(model_format, str) or model_format not in SUPPORTED_MODEL_FORMATS:
        issue(
            "invalid_input",
            f"不支持的 model_format: {model_format!r}; "
            f"可选: {list(SUPPORTED_MODEL_FORMATS)}",
        )

    if model is None:
        issue("invalid_input", "model 不能为空")

    if model_format in PATH_REQUIRED_FORMATS:
        if not isinstance(model, (str, os.PathLike)):
            issue(
                "invalid_input",
                f"model_format={model_format!r} 时, model 需为文件/目录路径",
            )
        checker = os.path.isfile if model_format == "seer_ir" else os.path.exists
        if not checker(model):
            issue("invalid_input", f"模型路径不存在: {model}")
    elif model_format == "onnx" and isinstance(model, (str, os.PathLike)):
        if not os.path.exists(model):
            issue("invalid_input", f"ONNX 模型路径不存在: {model}")

    # data
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        issue("invalid_input", f"batch_size 需为正整数, 收到: {batch_size!r}")

    if model_format in DATA_DICT_REQUIRED_FORMATS:
        _validate_input_spec(data_dict)


def validate_static_shapes(model: "onnx.ModelProto") -> None:
    """拒绝动态维度: ONNX 图中带形状的张量, 每一维都必须是静态的 dim_value。

    动态维 (dim_param) 或未知维会让 numel 等下游计算无法进行, 故入口直接拒绝。
    """
    bad: List[str] = []

    def _check(tensor_type, name: str) -> None:
        if not tensor_type.HasField("shape"):
            return
        for i, dim in enumerate(tensor_type.shape.dim):
            if not dim.HasField("dim_value"):
                bad.append(f"{name}[dim {i}] = {dim.dim_param or '?'}")

    for value_info in (
        list(model.graph.input)
        + list(model.graph.output)
        + list(model.graph.value_info)
    ):
        if value_info.type.HasField("tensor_type"):
            _check(value_info.type.tensor_type, value_info.name)

    if bad:
        preview = ", ".join(bad[:5])
        more = f" (共 {len(bad)} 处)" if len(bad) > 5 else ""
        issue(
            "dynamic_shape",
            "检测到动态/未知维度, ModelPerf 需要全静态形状才能计算代价: "
            f"{preview}{more}",
        )
