"""ModelPerf 统一异常层级。

所有面向用户的错误都继承自 :class:`ModelPerfError`，调用方（CLI / 上层脚本）
可一次性捕获并给出统一提示；同时保留语义子类，便于按类型分别处理。

    from modelperf.errors import ModelPerfError

    try:
        build_computation_graph(...)
    except ModelPerfError as exc:
        ...

输入类与算子类异常同时继承 ``ValueError``，以兼容按内置类型捕获的旧代码；
``HardwareError`` 仅继承基类，避免 ``KeyError.__str__`` 给消息加引号。
"""

from __future__ import annotations


class ModelPerfError(Exception):
    """ModelPerf 所有异常的基类。"""


class InvalidInputError(ModelPerfError, ValueError):
    """入口输入非法：模型 / 示例数据 / batch_size / 设备名等。"""


class DynamicShapeError(InvalidInputError):
    """图中存在动态或未知维度，无法进行静态代价计算。"""


class UnsupportedOpError(ModelPerfError, ValueError):
    """遇到未建模或不支持的算子类型。"""


class HardwareError(ModelPerfError):
    """硬件配置本身缺失或非法（如包内 hardware.yaml 不存在）。属系统/部署问题, 不可降级。"""


class UnknownDeviceError(HardwareError):
    """指定设备在硬件配置中未定义。属可降级情形: 跳过该设备的代价注入。"""
