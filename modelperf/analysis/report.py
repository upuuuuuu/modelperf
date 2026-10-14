"""统一入口: 分析并可视化计算图。"""

from __future__ import annotations

import os

from .graph_analysis import analyze_graph
from .graph_visualize import visualize_graph
from .console import warning_summary


def _ensure_parent_dir(path: str | None) -> None:
    """若 *path* 的父目录不存在则创建。"""
    if path:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)


def report(
    cg: "networkx.DiGraph",
    detailed: bool = False,
    csv_path: str | None = None,
    show_viewer: bool = False,
    port: int = 0,
):
    """对 *cg* 执行分析, 和/或交互式可视化。

    参数
    ----
    cg : networkx.DiGraph
        带节点/边 ``feature`` 属性的计算图。
    detailed : bool
        是否额外输出计算键、内存键与形状键。
    csv_path : str | None
        给定则把节点级 DataFrame 写入该 CSV 文件。
    show_viewer : bool
        是否启动交互式 Web viewer。
    port : int
        Web viewer 端口 (0 = 自动选择)。
    """
    _ensure_parent_dir(csv_path)

    # 报告前统一汇总构建期警告 (未知算子 / 缺形状等), 避免散落在构建过程中
    warning_summary(cg.graph.get("diagnostics"))

    stats = analyze_graph(cg, detailed=detailed, csv_path=csv_path)

    if show_viewer:
        # DataFrame 无法 JSON 序列化, viewer 侧转 records
        stats_serializable = {k: v.to_dict(orient="records") for k, v in stats.items()}
        visualize_graph(cg, stats=stats_serializable, port=port)
    return stats
