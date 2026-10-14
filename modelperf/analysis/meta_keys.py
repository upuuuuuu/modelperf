import re

# base keys
base_keys = ["id", "type"]

# computation_info
comp_keys_base = [
    "flops",
    "alu",
    "sfu",
]

comp_keys_extra = [
    "add",
    "sub",
    "mul",
    "div",
    "fma",
    "exp",
    "log",
    "pow",
    "sqrt",
    "trig",
    "hyperbolic",
    "logic",
    "round",
    "sign",
    "rcp",
]

# memory_info
mem_keys_base = [
    "mac",
    "read_act",
    "read_weight",
    "write_act",
]

mem_keys_extra = [
    "mac_wtemp",
    "read_temp",
    "read_act_weight",
    "read_act_weight_temp",
    "write_temp",
    "write_act_temp",
]

# shape_info
shape_keys_base = []

shape_keys_extra = [
    "in_tensor_infos",
    "out_tensor_infos",
    "p_tensor_infos",
]

# cost_info (roofline 代价, 单位秒); 实际键带设备后缀, 由 cost_keys() 展开
cost_keys_base = [
    "serial_cost",
    "roofline_cost",
]


# ── Hardware-suffixed keys ──────────────────────────────────────────
# cost_info 键带设备后缀, 如 serial_cost_hw_rtx3090
_HW_RE = re.compile(r"^(.+)_hw_([A-Za-z0-9_]+)$")


def split_hw_key(k):
    """serial_cost_hw_rtx3090 → ("serial_cost", "rtx3090"); 非硬件后缀键 → None。"""
    m = _HW_RE.match(str(k))
    return (m.group(1), m.group(2)) if m else None


def cost_keys(devices):
    """按设备列表展开 cost_info 键。

    cost_keys(["rtx3090"]) → ["serial_cost_hw_rtx3090", "roofline_cost_hw_rtx3090"]
    """
    return [f"{k}_hw_{dev}" for dev in devices for k in cost_keys_base]


# ── Pretty display labels ────────────────────────────────────────────
# 仅 *展示* 层（控制台表格 + CSV 表头）做美化。
# build_computation_graph 产出的底层数据键保持原样,
# 因此改动不触及 report 路径以外的任何地方。
PRETTY_LABELS = {
    # base
    "id": "ID",
    "type": "Type",
    "op": "Operation",
    "op_type": "Op Type",
    # computation
    "flops": "FLOPs",
    "alu": "ALU",
    "sfu": "SFU",
    "fma": "FMA",
    "add": "Add",
    "sub": "Sub",
    "mul": "Mul",
    "div": "Div",
    "exp": "Exp",
    "log": "Log",
    "pow": "Pow",
    "sqrt": "Sqrt",
    "trig": "Trig",
    "hyperbolic": "Hyperbolic",
    "logic": "Logic",
    "round": "Round",
    "sign": "Sign",
    "rcp": "Rcp",
    # memory
    "mac": "MAC",
    "read_act": "Read Activation",
    "read_weight": "Read Weight",
    "write_act": "Write Activation",
    "mac_wtemp": "MAC w/ Temp",
    "read_temp": "Read Temp",
    "read_act_weight": "Read Activation + Weight",
    "read_act_weight_temp": "Read Activation + Weight + Temp",
    "write_temp": "Write Temp",
    "write_act_temp": "Write Activation + Temp",
    # cost (roofline 代价, 单位: 秒)
    "serial_cost": "Serial Cost (s)",
    "roofline_cost": "Roofline Cost (s)",
    # shape
    "in_tensor_infos": "Input Tensors",
    "out_tensor_infos": "Output Tensors",
    "p_tensor_infos": "Param Tensors",
}


def pretty_key(k) -> str:
    """返回原始指标键 *k* 的友好展示名。

    硬件后缀键显示为 "Serial Cost (s) [rtx3090]"; 未知键回退为
    下划线转空格 + Title Case, 保证报告中不会漏出任何下划线。
    """
    s = split_hw_key(k)
    if s:
        base, dev = s
        return f"{pretty_key(base)} [{dev}]"

    if k in PRETTY_LABELS:
        return PRETTY_LABELS[k]
    return str(k).replace("_", " ").title()
