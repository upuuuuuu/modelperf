"""硬件参数加载与 roofline 代价计算。

硬件参数来源有二:
  1. 包内 hardware.yaml (devices 表, 每设备 peak_flops 按精度给出);
  2. 调用方直接传入设备信息 dict (此时完全不读 yaml)。
传入设备名得到该设备全部硬件参数; cost 计算时再按节点 dtype 取对应峰值算力。
"""

from pathlib import Path

import yaml

from .diagnostics import issue
from .errors import HardwareError, UnknownDeviceError

HARDWARE_YAML = Path(__file__).resolve().parent / "hardware.yaml"

_hardware_cache = None  # hardware.yaml 解析结果, 进程内只读一次
_hw_cache = {}  # 设备名 → 已规范化的硬件参数字典


def _load_hardware_yaml():
    """读取并缓存 hardware.yaml (进程内只解析一次)。

    hardware.yaml 属包内资源: 缺失或为空时**不抛异常**, 仅发 missing_hardware
    告警并返回空表 —— 没有硬件参数只是算不出 cost, 不致命, 交由上层 (inject_cost)
    跳过代价注入即可。
    """
    global _hardware_cache
    if _hardware_cache is None:
        if not HARDWARE_YAML.is_file():
            issue(
                "missing_hardware",
                f"未找到硬件配置文件 hardware.yaml: {HARDWARE_YAML}; 已跳过代价注入",
            )
            _hardware_cache = {}
        else:
            with open(HARDWARE_YAML, "r") as f:
                data = yaml.safe_load(f)
            if not data:
                issue(
                    "missing_hardware",
                    f"硬件配置文件为空或内容非法: {HARDWARE_YAML}; 已跳过代价注入",
                )
                _hardware_cache = {}
            else:
                _hardware_cache = data
    return _hardware_cache


def _normalize_hw(raw: dict) -> dict:
    """把原始设备参数规范化为统一结构 (yaml 路径与直接传 dict 共用本函数)。

    必需字段 device / peak_flops / peak_bytes; l2_bytes / l1_bytes 可缺省为 0。
    返回全新字典, 不持有入参引用, 避免调用方修改污染缓存。
    """
    try:
        device = str(raw["device"])
        peak_flops = {str(k): float(v) for k, v in raw["peak_flops"].items()}
        peak_bytes = float(raw["peak_bytes"])
    except (KeyError, AttributeError, TypeError, ValueError) as e:
        raise HardwareError(
            f"设备信息缺少必需字段或类型非法 (需 device / peak_flops / peak_bytes): {e}"
        )

    if not peak_flops:
        raise HardwareError(f"设备 '{device}' 的 peak_flops 不能为空")

    default_dtype = raw.get("default_dtype")
    if default_dtype not in peak_flops:
        default_dtype = next(iter(peak_flops))

    return dict(
        device=device,
        peak_flops=peak_flops,
        peak_bytes=peak_bytes,
        l2_bytes=int(raw.get("l2_bytes") or 0),
        l1_bytes=int(raw.get("l1_bytes") or 0),
        default_dtype=default_dtype,
    )


def _copy_hw(hw: dict) -> dict:
    """返回设备参数的独立副本 (peak_flops 一并复制), 供调用方安全修改。"""
    return {**hw, "peak_flops": dict(hw["peak_flops"])}


def load_hw(device=None):
    """读取设备硬件参数 (结果按设备名缓存)。

    参数
    ----
    device
        - ``None``: 取 hardware.yaml 的 ``default_device``;
        - ``str``: hardware.yaml 中的设备名, 未定义时抛 :class:`UnknownDeviceError`;
        - ``dict``: 直接给出设备信息 (``device`` / ``peak_flops`` / ``peak_bytes``,
          可选 ``l2_bytes`` / ``l1_bytes`` / ``default_dtype``), 此时**不读 hardware.yaml**。

    返回
    ----
    dict
        {device, peak_flops: {dtype: OP/s}, peak_bytes, l2_bytes, l1_bytes, default_dtype}
    """
    # 直接给设备信息: 完全不读 yaml
    if isinstance(device, dict):
        return _normalize_hw(device)

    hardware = _load_hardware_yaml()
    devices = hardware.get("devices") or {}

    if device is None:
        # hardware.yaml 缺失/为空时 default_device 一并缺失, 视为设备不可用
        device = hardware.get("default_device")
        if device is None:
            raise UnknownDeviceError(
                "hardware.yaml 未提供 default_device (文件缺失或为空), 无法取默认设备"
            )
    if device in _hw_cache:
        return _copy_hw(_hw_cache[device])
    if device not in devices:
        raise UnknownDeviceError(
            f"hardware.yaml 中未定义设备 '{device}', 可选: {list(devices)}"
        )

    hw = _normalize_hw(
        {
            "device": device,
            "default_dtype": hardware.get("default_dtype"),
            **devices[device],
        }
    )
    _hw_cache[device] = hw
    return _copy_hw(hw)


def list_devices():
    """返回 hardware.yaml 中定义的全部设备名 (文件缺失/为空时返回空列表)。"""
    return list((_load_hardware_yaml().get("devices") or {}))


def calc_cost(flops, mac, hw, dtype):
    """flops/mac → roofline 代价（单位: 秒）, 峰值算力按节点 *dtype* 取。

    dtype 不在设备精度表（如 int64/bool）时回退默认精度（fp32）。

    serial_cost   = flops/peak_flops + mac/peak_bytes     计算访存不重叠（上界）
    roofline_cost = max(flops/peak_flops, mac/peak_bytes) 屋顶线（下界）
    """
    peak_flops = hw["peak_flops"].get(dtype, hw["peak_flops"][hw["default_dtype"]])
    c = flops / peak_flops
    m = mac / hw["peak_bytes"]

    return {
        "serial_cost": c + m,
        "roofline_cost": max(c, m),
    }
