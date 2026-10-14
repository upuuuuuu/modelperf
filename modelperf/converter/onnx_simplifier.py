from onnxsim import simplify

from ..diagnostics import issue


def onnx_simplifier(onnx_model, optimize=True):
    """
    简化 ONNX 模型 (强制): onnxsim.simplify 内含常量折叠 + 形状推理,
    并可按 optimize 追加内部图优化。

    形状推理是下游建模的硬依赖 (中间张量的 shape 写进 value_info),
    因此不提供关闭选项。

    参数:
        onnx_model: onnx.ModelProto
        optimize: 是否执行内部图优化 (perform_optimization)
    返回:
        onnx.ModelProto
    """
    onnx_model, check = simplify(onnx_model, perform_optimization=optimize)
    if not check:
        issue("simplify_failed", "Simplification check failed")

    return onnx_model
