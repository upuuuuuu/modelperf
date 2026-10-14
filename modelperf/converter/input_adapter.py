from typing import Dict, List, Tuple, Any

from ..diagnostics import issue
from ..validation import SUPPORTED_DTYPES

# dtype 字符串 → 框架 dtype 属性 (torch/tf 的 dtype 名与白名单一致)
# bfloat16 无 numpy 等价类型, 在 numpy 分支单独回退
_NUMPY_DTYPE = {
    name: ("float32" if name == "bfloat16" else name) for name in SUPPORTED_DTYPES
}


# -------- PyTorch --------
def adapt_torch_kwargs(data_dict: Any, device: str = "cpu"):
    """
    输入可为：
      - 形状字典: {"x": {"shape": (N,D), "dtype": "float32"}, ...}
      - 已就绪:   {"x": torch.Tensor, ...}  -> 直接返回
    返回：dict[name, torch.Tensor]
    """
    try:
        import torch
    except Exception:
        raise

    # 已就绪：dict 的 value 是 Tensor
    if (
        isinstance(data_dict, dict)
        and data_dict
        and all(hasattr(v, "detach") for v in data_dict.values())
    ):
        return data_dict

    # 形状字典：造零张量
    str2dtype = {name: getattr(torch, name) for name in SUPPORTED_DTYPES}
    return {
        name: torch.zeros(spec["shape"], dtype=str2dtype[spec["dtype"]], device=device)
        for name, spec in (data_dict or {}).items()
    }


# -------- TensorFlow / Keras --------
def adapt_tf_signature(data_dict: Any):
    """
    输入可为：
      - list[tf.TensorSpec]   -> 直接返回
      - dict[name, Tensor]    -> 按张量生成 TensorSpec
      - 形状字典               -> 按 shape/dtype 生成 TensorSpec
    返回：list[tf.TensorSpec]
    """
    import tensorflow as tf

    # 已就绪：list / tuple of TensorSpec
    if (
        isinstance(data_dict, (list, tuple))
        and data_dict
        and isinstance(data_dict[0], tf.TensorSpec)
    ):
        return list(data_dict)

    specs = []
    if isinstance(data_dict, dict) and data_dict:
        any_val = next(iter(data_dict.values()))
        # dict[name, Tensor]
        if hasattr(any_val, "shape") and not isinstance(any_val, dict):
            for name, t in data_dict.items():
                t = tf.convert_to_tensor(t)
                specs.append(tf.TensorSpec(shape=t.shape, dtype=t.dtype, name=name))
            return specs
        # 形状字典
        str2dtype = {name: getattr(tf, name) for name in SUPPORTED_DTYPES}
        for name, spec in data_dict.items():
            specs.append(
                tf.TensorSpec(
                    shape=spec["shape"], dtype=str2dtype[spec["dtype"]], name=name
                )
            )
    return specs


# -------- MXNet --------
def adapt_mxnet_shapes_and_names(data_dict: Any) -> Tuple[List[Tuple], List[str]]:
    """
    输入可为：
      - 形状字典: {"x": {"shape": (..), "dtype": ...}, ...}
      - 简化形状: {"x": (..), "y": (..)}  -> 也支持
    返回：(input_shapes, input_names)
    """
    names = list(data_dict.keys())
    first_val = data_dict[names[0]]
    if isinstance(first_val, dict):  # 形状字典
        shapes = [tuple(spec["shape"]) for spec in data_dict.values()]
    else:  # 简化形状映射
        shapes = [tuple(v) for v in data_dict.values()]
    return shapes, names


# -------- Paddle --------
def adapt_paddle_shape_map(data_dict: Any):
    """
    Paddle 静态图导出通常不强依赖输入，但保留映射：
      - 形状字典 or 简化形状 -> {name: shape}
    """
    names = list(data_dict.keys())
    first_val = data_dict[names[0]]
    if isinstance(first_val, dict):
        return {n: tuple(spec["shape"]) for n, spec in data_dict.items()}
    return {n: tuple(shape) for n, shape in data_dict.items()}


# -------- Chainer（占位用）--------
def adapt_numpy_zeros(data_dict: Any):
    """
    返回 list[np.zeros(shape, dtype)]，给 Chainer 导出做占位
    """
    import numpy as np

    xs = []
    for _, spec in data_dict.items():
        if isinstance(spec, dict):
            shape, dtype = spec["shape"], spec["dtype"]
        else:
            shape, dtype = spec, "float32"
        xs.append(np.zeros(shape, dtype=_NUMPY_DTYPE.get(str(dtype), "float32")))
    return xs


# -------- JAX --------
def adapt_jax_inputs(data_dict: Any) -> List[Any]:
    """形状字典 -> list[np.ndarray], 作为 jax2onnx.to_onnx 的示例输入。"""
    return adapt_numpy_zeros(data_dict)


# -------- MindSpore --------
def adapt_mindspore_inputs(data_dict: Any) -> List[Any]:
    """形状字典 -> list[mindspore.Tensor], 作为 ms.export 的示例输入。"""
    import numpy as np
    import mindspore as ms

    # mindspore 各版本 dtype 覆盖不同, 按需解析而非预先构建整张映射表
    # (预建表时任一 getattr 失败都会导致整体报错)。
    return [
        ms.Tensor(
            np.zeros(spec["shape"], dtype=_NUMPY_DTYPE[spec["dtype"]]),
            getattr(ms, spec["dtype"]),
        )
        for spec in data_dict.values()
    ]


# -------- OneFlow --------
def adapt_oneflow_inputs(data_dict: Any) -> List[Any]:
    """形状字典 -> list[flow.Tensor], 供 nn.Graph._compile 生成静态图。"""
    import oneflow as flow

    # oneflow 无 int16 等 dtype, 故按需解析而非预先构建整张映射表
    return [
        flow.zeros(spec["shape"], dtype=getattr(flow, spec["dtype"]))
        for spec in data_dict.values()
    ]


def adapt_input(mode: str, data_dict: Any):
    if mode == "onnx":
        return data_dict
    if mode == "torch":
        return adapt_torch_kwargs(data_dict)
    if mode in ("tensorflow", "keras"):
        return adapt_tf_signature(data_dict)
    if mode == "mxnet":
        return adapt_mxnet_shapes_and_names(data_dict)
    if mode == "paddle":
        return adapt_paddle_shape_map(data_dict)
    if mode == "chainer":
        return adapt_numpy_zeros(data_dict)
    if mode == "jax":
        return adapt_jax_inputs(data_dict)
    if mode == "mindspore":
        return adapt_mindspore_inputs(data_dict)
    if mode == "oneflow":
        return adapt_oneflow_inputs(data_dict)
    issue("invalid_input", f"Unknown mode: {mode}")
