"""分析包共享的通用小工具: 类型转换与张量信息提取。"""

from __future__ import annotations

from typing import Any


# ── Type coercion ────────────────────────────────────────────────────
def as_int(x: Any, default: int = 0) -> int:
    """安全地把 *x* 转为 int, 失败时返回 *default*。"""
    try:
        return int(x) if x is not None else default
    except (ValueError, TypeError):
        return default


def as_float(x: Any, default: float = 0.0) -> float:
    """安全地把 *x* 转为 float, 失败时返回 *default*。"""
    try:
        return float(x) if x is not None else default
    except (ValueError, TypeError):
        return default


def as_str(x: Any, default: str = "") -> str:
    """安全地把 *x* 转为 str, 失败时返回 *default*。"""
    try:
        return str(x) if x is not None else default
    except Exception:
        return default


def fmt_value(val: Any, fmt: str) -> str:
    """按 *fmt* ('int' / 'float' / 'repr') 格式化 *val*。"""
    if fmt == "int":
        return f"{as_int(val):,}"
    if fmt == "float":
        return f"{as_float(val):.6g}"
    return as_str(val)


# ── Tensor info ──────────────────────────────────────────────────────
def shapes(infos: list[dict[str, Any]] | None) -> list[tuple[Any, Any]]:
    """从 tensor-info 字典列表提取 (shape, dtype_size) 列表。

    非 dict 元素 (理论上不该出现) 以 ``(None, None)`` 占位, 保持与输入等长,
    避免上层按张量下标对齐时错位。
    """
    if not infos:
        return []
    return [
        (t.get("shape"), t.get("dtype_size")) if isinstance(t, dict) else (None, None)
        for t in infos
    ]
