import time
import pickle

from .converter import adapt_input, normalize_qdq, onnx_converter, onnx_simplifier
from .graph import build_ir, build_cg
from .cost import inject_cost
from .diagnostics import collect_diagnostics
from .errors import InvalidInputError
from .validation import (
    DATA_DICT_REQUIRED_FORMATS,
    validate_build_inputs,
    validate_static_shapes,
)

from .analysis.console import logo, step


def build_computation_graph(
    model=None,
    model_format="onnx",
    data_dict=None,
    batch_size=1,
    devices=None,
    optimize=True,
    opset=16,
):
    """构建计算图 (CG): 模型 -> ONNX -> IR -> CG, 并按设备注入 roofline 代价。

    参数
    ----
        model: 模型对象, 或模型路径 (onnx / tf_savedmodel / tf_graphdef / tflite /
            paddle / caffe / seer_ir 等路径型格式)。
        model_format: 输入模型格式, 取值见 validation.SUPPORTED_MODEL_FORMATS。
        data_dict: 形状字典 {输入名: {"shape": (...), "dtype": "..."}}, 仅
            validation.DATA_DICT_REQUIRED_FORMATS 中的格式需要。
        batch_size: 批大小, 默认 1。**仅作图级元数据** (cg.graph["batch_size"]),
            供可视化展示, 不参与 shape / FLOPs / MAC / 代价任何计算; 真实 batch 维
            由模型本身或 data_dict 决定。故无需按批大小调参, 保持默认即可, 只有刻意
            想在图标注其它批大小时才显式传; 传则须为正整数。
        devices: 目标硬件, 可为单个 device 名 / 设备信息 dict / 其列表; None /
            未定义设备名 / hardware.yaml 缺失或为空, 均仅告警并跳过, 不注入对应代价。
            传 dict 时直接使用该设备信息, 不读 hardware.yaml。
        optimize: onnxsim 简化时是否顺带做 ONNX optimizer 图优化 (不影响强制执行的
            常量折叠与形状推理)。
        opset: 导出 ONNX 使用的 opset 版本。
    """
    logo()
    validate_build_inputs(model, model_format, data_dict, batch_size)

    # 构建全程收集诊断 (未知算子 / 缺形状等降级警告), 供 report 前统一汇总
    with collect_diagnostics() as diagnostics:
        # ir builder
        if model_format == "seer_ir":
            try:
                with open(model, "rb") as f:
                    ir = pickle.load(f)
            except (
                OSError,
                EOFError,
                pickle.UnpicklingError,
                AttributeError,
                ImportError,
            ) as e:
                raise InvalidInputError(f"seer_ir 模型加载失败: {model!r} ({e})") from e
        else:
            # 规范形状字典 -> 各后端原生形态 (由 ModelPerf 内部构造示例张量)
            if model_format in DATA_DICT_REQUIRED_FORMATS:
                data_dict = adapt_input(model_format, data_dict)

            # any2onnx
            st = time.time()
            onnx_model = onnx_converter(model, model_format, data_dict, opset)
            et = time.time()
            step("any2onnx", et - st)

            # onnx simplifier
            st = time.time()
            onnx_model = onnx_simplifier(onnx_model, optimize)
            et = time.time()
            step("onnx simplify", et - st)

            # 量化图归一化: 折叠 QDQ 表示 (必须在 shape inference 之后, 否则
            # 拿不到 Q 输出的 INT8 声明); 非量化图零副作用, 不打印该步骤
            st = time.time()
            onnx_model, normalized = normalize_qdq(onnx_model)
            et = time.time()
            if normalized:
                step("qdq normalize", et - st)

            # 动态维度入口拒绝: 代价计算要求全静态形状
            validate_static_shapes(onnx_model)

            st = time.time()
            ir = build_ir(onnx_model, batch_size)
            et = time.time()
            step("ir builder", et - st)

        # cg builder
        st = time.time()
        cg = build_cg(ir)
        et = time.time()
        step("cg builder", et - st)

        # cost inject: 按设备注入 roofline 代价 (None / 未知设备由 inject_cost 内部处理)
        inject_cost(cg, devices)

    # 构建期诊断挂到 cg.graph, 便于 report / 调用方统一取用
    cg.graph["diagnostics"] = [d.to_dict() for d in diagnostics]

    return ir, cg
