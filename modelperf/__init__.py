"""ModelPerf — 深度学习模型性能分析器。

把任意模型转成 ONNX, 构建带 roofline 代价的计算图, 并在交互式网页里查看::

    import modelperf

    ir, cg = modelperf.build_computation_graph(
        model=model, model_format="torch", data_dict=data, devices=["rtx3090"]
    )
    modelperf.report(cg)                       # 终端统计
    modelperf.export_html(cg, "graph.html")    # 单文件 HTML
    modelperf.visualize_graph(cg)              # 本地交互式 viewer
"""

from .cg_builder import build_computation_graph
from .cost import inject_cost
from .diagnostics import (
    ModelPerfWarning,
    Severity,
    collect_diagnostics,
    configure,
    reset,
)
from .errors import (
    DynamicShapeError,
    HardwareError,
    InvalidInputError,
    ModelPerfError,
    UnknownDeviceError,
    UnsupportedOpError,
)
from .hardware import calc_cost, list_devices, load_hw
from .analysis import (
    GraphWebViewer,
    analyze_graph,
    export_html,
    report,
    visualize_graph,
)

__version__ = "0.1.0"

__all__ = [
    "build_computation_graph",
    "inject_cost",
    "report",
    "analyze_graph",
    "visualize_graph",
    "export_html",
    "GraphWebViewer",
    "load_hw",
    "list_devices",
    "calc_cost",
    "ModelPerfError",
    "InvalidInputError",
    "DynamicShapeError",
    "UnsupportedOpError",
    "HardwareError",
    "UnknownDeviceError",
    "ModelPerfWarning",
    "Severity",
    "configure",
    "reset",
    "collect_diagnostics",
    "__version__",
]
