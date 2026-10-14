"""ModelPerf 统一诊断机制: 可配置严重级别 + 统一发声点 + 收集器。

设计目标
--------
同一类问题在不同场景下可能需要不同处置: 有的属于"能继续跑, 记录即可"
(如未知算子、缺失形状), 有的属于"跑不下去, 必须中断" (如动态形状、输入非法)。
本模块把"问题类型 -> 严重级别 / 异常类"收敛成一张策略表 (:data:`ISSUES`),
并暴露 :func:`configure` 作为超参数入口, 调用方可在构建前一次性调整
**何时警告、何时报错**; 所有发声点统一走 :func:`issue`, 不再散落
``raise`` / ``warnings.warn``。

用法
----
    from modelperf.diagnostics import configure, collect_diagnostics

    configure(unknown_op="error")          # 把未知算子提升为错误
    with collect_diagnostics() as diags:   # 构建期收集, 报告前统一汇总
        ir, cg = build_computation_graph(...)
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Type

from .errors import (
    DynamicShapeError,
    InvalidInputError,
    ModelPerfError,
    UnsupportedOpError,
)


class Severity(str, Enum):
    """问题的严重级别: 警告(继续) / 错误(中断)。"""

    WARN = "warn"
    ERROR = "error"


class ModelPerfWarning(UserWarning):
    """ModelPerf 自定义警告类别, 便于调用方单独过滤。"""


@dataclass(frozen=True)
class IssueSpec:
    """单个问题类型的默认策略。"""

    severity: Severity
    error_cls: Type[ModelPerfError] = ModelPerfError
    description: str = ""


# 策略表: 问题类型 -> (默认严重级别, 错误时抛出的异常类)
#   - WARN: 能继续算, 记录并警告 (代价按缺省值计)
#   - ERROR: 无法继续, 抛对应 ModelPerfError 子类
ISSUES: Dict[str, IssueSpec] = {
    # ---- 可降级: 记录并继续 ----
    "unknown_op": IssueSpec(Severity.WARN, UnsupportedOpError, "遇到未建模的算子类型"),
    "unknown_subop": IssueSpec(
        Severity.WARN, UnsupportedOpError, "算子子分派未覆盖的分支"
    ),
    "op_shape": IssueSpec(Severity.WARN, description="算子输入/输出形状不满足建模前提"),
    "missing_shape": IssueSpec(Severity.WARN, description="张量缺少形状信息"),
    "missing_dtype_size": IssueSpec(
        Severity.WARN, description="张量缺少 dtype 大小信息"
    ),
    "missing_arg": IssueSpec(Severity.WARN, description="算子属性缺失, 已回退默认值"),
    "unknown_device": IssueSpec(
        Severity.WARN, description="目标设备未定义, 跳过代价注入"
    ),
    "device_unspecified": IssueSpec(
        Severity.WARN, description="未指定目标设备, 跳过代价注入"
    ),
    "missing_hardware": IssueSpec(
        Severity.WARN, description="硬件配置缺失或为空, 跳过代价注入"
    ),
    "qdq_boundary": IssueSpec(
        Severity.WARN, description="量化图的 dtype 转换边界被保留 (岛入口 / 图 I/O)"
    ),
    "qdq_residual": IssueSpec(
        Severity.WARN, description="量化图归一化未覆盖的情形, 访存口径可能偏离真实执行"
    ),
    # ---- 中断: 抛异常 ----
    "missing_required_arg": IssueSpec(
        Severity.ERROR, ModelPerfError, "算子必填属性缺失"
    ),
    "invalid_input": IssueSpec(Severity.ERROR, InvalidInputError, "入口输入非法"),
    "dynamic_shape": IssueSpec(
        Severity.ERROR, DynamicShapeError, "图中存在动态/未知维度"
    ),
    "simplify_failed": IssueSpec(Severity.ERROR, ModelPerfError, "ONNX 简化校验失败"),
}


# 严重级别覆盖 (由 configure 设置); 未覆盖的走 ISSUES 默认值
_severity_override: Dict[str, Severity] = {}


def _normalize_severity(value) -> Severity:
    if isinstance(value, Severity):
        return value
    try:
        return Severity(str(value).lower())
    except ValueError:
        raise ValueError(f"非法严重级别: {value!r} (可选 'warn' / 'error')")


def configure(**overrides) -> None:
    """调整问题类型的全局严重级别 (超参数入口)。

    参数
    ----
    **overrides
        ``{issue_code: 'warn' | 'error'}``; 未知 code 会报错。
        例::

            configure(unknown_op="error", missing_shape="warn")
    """
    for code, value in overrides.items():
        if code not in ISSUES:
            raise ValueError(f"未知 issue code: {code!r}; 可选: {sorted(ISSUES)}")
        _severity_override[code] = _normalize_severity(value)


def reset() -> None:
    """清空所有严重级别覆盖, 恢复默认策略。"""
    _severity_override.clear()


def severity_of(code: str) -> Severity:
    """返回某问题类型当前生效的严重级别 (含覆盖)。"""
    if code in _severity_override:
        return _severity_override[code]
    if code not in ISSUES:
        raise ValueError(f"未知 issue code: {code!r}")
    return ISSUES[code].severity


@dataclass
class Diagnostic:
    """一条已发生的诊断记录。"""

    code: str
    message: str
    severity: Severity
    context: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "context": self.context,
        }


_collected: Optional[List[Diagnostic]] = None


@contextmanager
def collect_diagnostics():
    """在 ``with`` 块内收集所有 issue, 退出后可从块变量取回列表。

    支持嵌套: 内层块结束后恢复外层收集器。
    """
    global _collected
    prev = _collected
    collected: List[Diagnostic] = []
    _collected = collected
    try:
        yield collected
    finally:
        _collected = prev


def _record(diag: Diagnostic) -> None:
    if _collected is not None:
        _collected.append(diag)


def issue(code: str, message: str, **context):
    """统一发声点: 按当前策略警告并继续, 或抛出对应异常。

    参数
    ----
    code : str
        问题类型, 需在 :data:`ISSUES` 中登记。
    message : str
        人类可读描述。
    **context
        附加上下文 (算子名 / 张量名等), 一并记录进 :class:`Diagnostic`。
    """
    if code not in ISSUES:
        raise ValueError(f"未知 issue code: {code!r}; 可选: {sorted(ISSUES)}")

    spec = ISSUES[code]
    severity = severity_of(code)
    _record(Diagnostic(code, message, severity, dict(context)))

    if severity is Severity.ERROR:
        raise spec.error_cls(message)

    warnings.warn(f"[{code}] {message}", ModelPerfWarning, stacklevel=2)
