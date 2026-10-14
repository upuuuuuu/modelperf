"""代价注入: 按设备把 roofline 代价写进计算图。

流水线 ``... -> build_cg -> inject_cost`` 的最后一级独立阶段, 只依赖硬件层
(``hardware``) 与计算图的 feature schema, 不感知上层编排逻辑。
"""

import time

from .diagnostics import issue
from .errors import UnknownDeviceError
from .hardware import calc_cost, load_hw

from .analysis.console import step


def _node_dtype(feat, hw):
    """节点计算精度: 取主输入 dtype, 无输入回退主输出, 再回退设备默认精度。

    Cast / QuantizeLinear 等输入输出类型不一致的算子以输入为准
    （此类算子 flops/mac ≈ 0, 对 cost 影响可忽略）。
    """
    infos = feat["in_tensor_infos"] or feat["out_tensor_infos"]
    dtype = infos[0].get("dtype") if infos else None
    return dtype or hw["default_dtype"]


def inject_cost(cg, devices):
    """按设备注入 roofline 代价: 节点级 feat["cost_info"], 图级 cg.graph["cost_info"]。

    键加 _hw_<device> 后缀; 精度由各节点 tensor 类型决定; 图级 cost 为节点求和。

    devices 允许 None / 单个设备名或设备信息 dict / 其列表:
      - None            → 告警"未指定设备"并跳过, 不注入代价;
      - 未定义设备名    → 告警并跳过该设备, 不阻断流水线;
      - 设备信息 dict   → 直接使用, 不读 hardware.yaml。
    全部设备都不可用时整条代价注入跳过。
    """
    if devices is None:
        issue("device_unspecified", "未指定目标设备, 跳过 roofline 代价注入")
        return
    if isinstance(devices, (str, dict)):
        devices = [devices]

    # 设备不可用统一降级: 未定义设备 / hardware.yaml 缺失或为空均告警并跳过
    hws = []
    for dev in devices:
        try:
            hws.append(load_hw(dev))
        except UnknownDeviceError as e:
            issue("unknown_device", f"{e}; 已跳过该设备, 不注入其代价", device=dev)
    if not hws:
        return

    st = time.time()
    graph_cost = {
        hw["device"]: {"serial_cost": 0.0, "roofline_cost": 0.0} for hw in hws
    }

    for node_idx in cg.nodes:
        feat = cg.nodes[node_idx]["feature"]
        flops = feat["computation_info"]["flops"]
        mac = feat["memory_info"]["mac"]
        feat.setdefault("cost_info", {})

        for hw in hws:
            cost = calc_cost(flops, mac, hw, _node_dtype(feat, hw))
            feat["cost_info"].update(
                {f"{k}_hw_{hw['device']}": v for k, v in cost.items()}
            )
            for k, v in cost.items():
                graph_cost[hw["device"]][k] += v

    cg.graph["devices"] = [hw["device"] for hw in hws]
    cg.graph["cost_info"] = {
        f"{k}_hw_{dev}": v for dev, c in graph_cost.items() for k, v in c.items()
    }
    et = time.time()
    step(f"cost ({','.join(cg.graph['devices'])})", et - st)
