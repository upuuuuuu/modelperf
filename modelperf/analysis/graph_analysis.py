import os

import pandas as pd

from .meta_keys import (
    base_keys,
    comp_keys_base,
    comp_keys_extra,
    cost_keys,
    mem_keys_base,
    mem_keys_extra,
    pretty_key,
    shape_keys_base,
    shape_keys_extra,
    split_hw_key,
)
from .console import ok, print_table, section
from .utils import shapes


def _select_keys(detailed: bool):
    """按 *detailed* 标志选择要收集的键集合。"""
    comp_keys = comp_keys_base + comp_keys_extra if detailed else comp_keys_base
    mem_keys = mem_keys_base + mem_keys_extra if detailed else mem_keys_base
    shape_keys = shape_keys_base + shape_keys_extra if detailed else shape_keys_base
    return comp_keys, mem_keys, shape_keys


def collect_node_features(cg, detailed: bool = False) -> pd.DataFrame:
    """把所有节点 feature 收集为 DataFrame, 并附一行 ``total`` 汇总。

    cost_info（roofline 代价, 键带 _hw_<device> 后缀）为可选注入: 键由
    ``cg.graph['devices']`` 经 :func:`cost_keys` 静态展开, 与 base_keys 等同源;
    未注入代价时设备列表为空, 不产生 cost 列。
    """
    comp_keys, mem_keys, shape_keys = _select_keys(detailed)
    feats = [cg.nodes[n]["feature"] for n in cg.nodes]
    cost_key_list = cost_keys(cg.graph.get("devices") or [])

    base_dict = {k: [] for k in base_keys}
    comp_dict = {k: [] for k in comp_keys}
    mem_dict = {k: [] for k in mem_keys}
    shape_dict = {k: [] for k in shape_keys}
    cost_dict = {k: [] for k in cost_key_list}

    for feat in feats:
        for k in base_keys:
            base_dict[k].append(feat[k])
        for k in comp_keys:
            comp_dict[k].append(feat["computation_info"][k])
        for k in mem_keys:
            mem_dict[k].append(feat["memory_info"][k])
        for k in shape_keys:
            shape_dict[k].append(shapes(feat[k]))
        cost = feat.get("cost_info") or {}
        for k in cost_dict:
            cost_dict[k].append(cost.get(k, 0))

    df = pd.DataFrame({**base_dict, **comp_dict, **mem_dict, **cost_dict, **shape_dict})

    # Append a total row for numeric columns (逐列求和, 避免 int 列被整体上转 float)
    numeric_cols = df.select_dtypes(include="number").columns
    total_row = {
        **{c: df[c].sum() for c in numeric_cols},
        **{k: "total" for k in df.columns if k not in numeric_cols},
    }
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
    return df


def compute_graph_stats(cg) -> dict:
    """按节点类型与原子操作聚合节点统计。

    返回
    ----
    dict
        - ``typeAgg``: 每个节点类型 —— count, flops, mac, read_act, read_weight, write_act
        - ``opAgg``: 每种操作类型 (comp_keys_extra) —— count (节点数), ops (实例总数)
    """
    type_agg = {}
    op_agg = {}

    for node_idx in cg.nodes:
        feat = cg.nodes[node_idx]["feature"]
        comp = feat["computation_info"]
        mem = feat["memory_info"]

        if feat["type"] not in type_agg:
            type_agg[feat["type"]] = {
                "count": 0,
                "flops": 0,
                "mac": 0,
                "read_act": 0,
                "read_weight": 0,
                "write_act": 0,
            }
        t = type_agg[feat["type"]]
        t["count"] += 1
        t["flops"] += comp["flops"]
        t["mac"] += mem["mac"]
        t["read_act"] += mem["read_act"]
        t["read_weight"] += mem["read_weight"]
        t["write_act"] += mem["write_act"]

        for op in comp_keys_extra:
            v = comp.get(op, 0)
            if v > 0:
                if op not in op_agg:
                    op_agg[op] = {"count": 0, "ops": 0}
                op_agg[op]["count"] += 1
                op_agg[op]["ops"] += v

    type_df = pd.DataFrame(
        [
            {"type": k, **v}
            for k, v in sorted(type_agg.items(), key=lambda kv: -kv[1]["flops"])
        ]
    )
    op_df = pd.DataFrame(
        [
            {"op": k, **v}
            for k, v in sorted(op_agg.items(), key=lambda kv: -kv[1]["ops"])
        ]
    )
    return {"typeAgg": type_df, "opAgg": op_df}


def analyze_graph(cg, detailed: bool = False, csv_path: str | None = None) -> dict:
    """打印节点级 feature 与图统计, 可选写入 CSV。

    返回预计算的图统计 dict（供 web viewer 用）。
    展示/CSV 表头经 ``pretty_key`` 美化; 返回的 stats dict 保留原始键,
    因此不改动流水线其余部分。
    """
    df = collect_node_features(cg, detailed=detailed)
    stats = compute_graph_stats(cg)

    # 终端显示: cost 列（_hw_ 后缀）3 位有效数字; CSV 保留原始精度
    disp_df = df.copy()
    for k in disp_df.columns:
        if split_hw_key(k):
            disp_df[k] = disp_df[k].map(
                lambda v: f"{v:.3g}" if isinstance(v, float) else v
            )

    section("Node-level Features")
    print_table(disp_df.rename(columns=pretty_key).to_dict(orient="list"))

    type_df, op_df = stats["typeAgg"], stats["opAgg"]
    section("By Node Type (Count / FLOPs / MAC)")
    print_table(type_df.to_dict(orient="list"))
    section("By Operation Type (Count / Ops)")
    print_table(op_df.to_dict(orient="list"))

    if csv_path:
        parent = os.path.dirname(csv_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        df.rename(columns=pretty_key).to_csv(csv_path, index=False)
        ok(f"Saved node-level features to {csv_path}")

    return stats
