"""终端输出: ANSI 配色、品牌 logo 与步骤 / 表格等控制台 UI。"""

from __future__ import annotations

import os
import sys


# ── Console styling (ANSI color, auto-disabled when piped) ──────────
def _use_color() -> bool:
    """仅在 stdout 为 TTY 且未被禁用时启用 ANSI 配色。"""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR") or os.environ.get("CLICOLOR_FORCE"):
        return True
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


_USE = _use_color()


def _c(code: str, s: object) -> str:
    return f"\x1b[{code}m{s}\x1b[0m" if _USE else str(s)


def dim(s: object) -> str:
    return _c("90", s)


def bold(s: object) -> str:
    return _c("1", s)


def cyan(s: object) -> str:
    return _c("36", s)


def blue(s: object) -> str:
    return _c("34", s)


def green(s: object) -> str:
    return _c("32", s)


def yellow(s: object) -> str:
    return _c("33", s)


# ── Brand logo ───────────────────────────────────────────────────────
_LOGO = (
    "███╗   ███╗  ██████╗  ██████╗  ███████╗ ██╗      ██████╗  ███████╗ ██████╗  ███████╗",
    "████╗ ████║ ██╔═══██╗ ██╔══██╗ ██╔════╝ ██║      ██╔══██╗ ██╔════╝ ██╔══██╗ ██╔════╝",
    "██╔████╔██║ ██║   ██║ ██║  ██║ █████╗   ██║      ██████╔╝ █████╗   ██████╔╝ █████╗  ",
    "██║╚██╔╝██║ ██║   ██║ ██║  ██║ ██╔══╝   ██║      ██╔═══╝  ██╔══╝   ██╔══██╗ ██╔══╝  ",
    "██║ ╚═╝ ██║ ╚██████╔╝ ██████╔╝ ███████╗ ███████╗ ██║      ███████╗ ██║  ██║ ██║     ",
    "╚═╝     ╚═╝  ╚═════╝  ╚═════╝  ╚══════╝ ╚══════╝ ╚═╝      ╚══════╝ ╚═╝  ╚═╝ ╚═╝     ",
)
_LOGO_GRADIENT = ("96", "96", "36", "36", "34", "34")
_LOGO_SHOWN = False


def logo(force: bool = False) -> None:
    """打印 ModelPerf 品牌 logo（进程内仅一次; 遵循 NO_COLOR）。"""
    global _LOGO_SHOWN
    if _LOGO_SHOWN and not force:
        return
    _LOGO_SHOWN = True

    from .. import __version__

    width = max(len(line) for line in _LOGO)
    print()
    for code, line in zip(_LOGO_GRADIENT, _LOGO):
        print(_c(code, line))
    print(
        dim(
            f"A Deep Learning Model Performance Analyzer · v{__version__}".center(
                width
            )
        )
    )
    print()


def banner(title: str, subtitle: str | None = None, width: int = 72) -> None:
    """打印启动横幅。"""
    inner = width - 2

    def row(text: str, style=lambda s: s) -> str:
        pad = (inner - len(text)) // 2
        return (
            blue("║")
            + " " * pad
            + style(text)
            + " " * (inner - pad - len(text))
            + blue("║")
        )

    print(blue("═" * width))
    print(row(title, bold))
    if subtitle:
        print(row(subtitle, dim))
    print(blue("═" * width))


def section(title: str, width: int = 60) -> None:
    """在表格上方打印分节标题。"""
    print("\n" + bold(title))
    print(dim("─" * min(width, max(len(title) + 4, 8))))


def kv(key: object, value: object, key_w: int = 26) -> None:
    """打印一行对齐的键值对。"""
    print("  " + dim(str(key).ljust(key_w)) + bold(str(value)))


def step(name: str, seconds: float) -> None:
    """打印一个流水线步骤及其耗时。"""
    print(
        "  "
        + cyan("▶")
        + " "
        + dim(str(name).ljust(26))
        # 先补齐再着色: rjust 若作用于带 ANSI 转义的串会按转义符计宽而错位
        + green(f"{seconds:.2f}s".rjust(9))
    )


def ok(msg: str) -> None:
    """打印一条成功消息。"""
    print("  " + green("✔") + " " + dim(msg))


def warn(msg: str) -> None:
    """打印一条警告消息。"""
    print("  " + yellow("!") + " " + msg)


def warning_summary(diagnostics, limit: int = 10, title: str = "构建期警告") -> int:
    """汇总构建期收集到的诊断: 按类型归并计数 + 展示前 *limit* 条明细。

    参数
    ----
    diagnostics : list[dict] | None
        :func:`modelperf.diagnostics.collect_diagnostics` 收集到的诊断 (``to_dict`` 结果)。
    limit : int
        最多展示的明细条数, 超出部分仅提示数量。
    title : str
        分节标题。

    返回
    ----
    int
        诊断总条数 (0 表示无警告, 不打印任何内容)。
    """
    if not diagnostics:
        return 0

    from collections import Counter

    counts = Counter(d["code"] for d in diagnostics)
    section(f"{title} ({len(diagnostics)})")
    for code, n in counts.most_common():
        warn(f"{yellow(code)} ×{n}")
    for d in diagnostics[:limit]:
        print("    " + dim(f"[{d['code']}] {d['message']}"))
    if len(diagnostics) > limit:
        print("    " + dim(f"... 其余 {len(diagnostics) - limit} 条已省略"))
    return len(diagnostics)


def print_table(data: dict[str, list]) -> None:
    """按 {列名: [值]} 打印带框线表格。

    首列等于 ``total`` 的末行会渲染为加粗汇总行（与 ``collect_node_features`` 对齐）。
    """
    header = list(data.keys())
    if not header:
        return
    n_rows = len(next(iter(data.values())))
    rows = [
        {k: ("" if data[k][i] is None else str(data[k][i])) for k in header}
        for i in range(n_rows)
    ]

    widths = {}
    for i, h in enumerate(header):
        widths[i] = max([len(h)] + [len(r[h]) for r in rows]) + 2

    def sep(left: str, mid: str, right: str) -> str:
        return left + mid.join("─" * widths[i] for i in range(len(header))) + right

    def cell(v: str, i: int, style=lambda s: s) -> str:
        return " " + style(f"{v:<{widths[i] - 2}}") + " "

    print(sep("┌", "┬", "┐"))
    print("│" + "│".join(cell(h, i, bold) for i, h in enumerate(header)) + "│")
    print(sep("├", "┼", "┤"))
    for idx, r in enumerate(rows):
        is_total = str(r[header[0]]).lower() == "total"
        style = bold if is_total else (lambda s: s)
        print("│" + "│".join(cell(r[k], i, style) for i, k in enumerate(header)) + "│")
        if is_total and idx < len(rows) - 1:
            print(sep("├", "┼", "┤"))
    print(sep("└", "┴", "┘"))
