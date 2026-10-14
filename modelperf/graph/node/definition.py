class Node:
    def __init__(self):
        self.id = ""
        self.type = ""
        self.args = {}

        self.in_tensor_infos = []
        self.p_tensor_infos = []
        self.out_tensor_infos = []

        self.computation_info = ComputationInfo()
        self.memory_info = MemoryAccessInfo()

    def to_dict_ir(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "args": self.args,
            "in_tensor_infos": self.in_tensor_infos,
            "p_tensor_infos": self.p_tensor_infos,
            "out_tensor_infos": self.out_tensor_infos,
        }

    def to_dict_cg(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "args": self.args,
            "in_tensor_infos": self.in_tensor_infos,
            "p_tensor_infos": self.p_tensor_infos,
            "out_tensor_infos": self.out_tensor_infos,
            "computation_info": self.computation_info.to_dict(),
            "memory_info": self.memory_info.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        obj = cls()
        obj.id = d.get("id", "")
        obj.type = d.get("type", "")
        obj.args = d.get("args", {})
        obj.in_tensor_infos = d.get("in_tensor_infos", [])
        obj.out_tensor_infos = d.get("out_tensor_infos", [])
        obj.p_tensor_infos = d.get("p_tensor_infos", [])
        # CG 字典带 computation_info / memory_info; 不还原会丢失算子计数
        if "computation_info" in d:
            obj.computation_info = ComputationInfo.from_dict(d["computation_info"])
        if "memory_info" in d:
            obj.memory_info = MemoryAccessInfo.from_dict(d["memory_info"])
        return obj


class ComputationInfo:
    """操作计数统计类（原始操作计数）"""

    def __init__(self):
        """初始化计算复杂度统计计数器"""
        # 基本算术运算
        self.add = 0  # 加法操作次数 (a + b)
        self.sub = 0  # 减法操作次数 (a - b)
        self.mul = 0  # 乘法操作次数 (a * b)
        self.div = 0  # 除法操作次数 (a / b) # NOTE
        self.fma = 0  # 乘加融合操作次数 (a * b + c)

        # 超越函数运算
        self.pow = 0  # 幂运算次数 pow(a, b) = exp(b⋅lna)
        self.sqrt = 0  # 平方根操作次数 (sqrt/rsqrt)
        self.exp = 0  # 指数操作次数 (e^x)
        self.log = 0  # 自然对数操作次数 (ln(x))
        self.trig = 0  # 三角函数操作次数 (sin/cos/tan/arcsin/arccos/arctan等)
        self.hyperbolic = 0  # 双曲函数操作次数 (sinh/cosh/tanh/asinh/acosh/atanh等)

        # 其他运算
        self.logic = 0  # 比较/逻辑/mask类操作（cmp, logic, select）
        self.round = 0  # 舍入类（round/floor/ceil/trunc）
        self.sign = 0  # 符号位类操作（abs/neg）
        self.rcp = 0  # 倒数及类似（1/a）

    @property
    def flops(self) -> int:
        """
        传统FLOPs计算
        """
        return (
            #
            self.add
            + self.sub
            + self.mul
            + self.div
            + self.fma * 2
            #
            + self.pow
            + self.sqrt
            + self.exp
            + self.log
            + self.trig
            + self.hyperbolic
            #
            + self.logic
            + self.round
            + self.sign
            + self.rcp
        )

    @property
    def alu(self) -> int:
        """
        计算ALU（Arithmetic Logic Unit）操作次数
        """
        return (
            self.add
            + self.sub
            + self.mul
            + self.fma
            + self.logic
            + self.sign
            + self.round
        )

    @property
    def sfu(self) -> int:
        """
        计算SFU（Special Function Unit）操作次数
        """
        return (
            self.div
            + self.rcp
            + self.sqrt
            + self.pow
            + self.exp
            + self.log
            + self.trig
            + self.hyperbolic
        )

    def to_dict(self) -> dict:
        data = {
            "flops": self.flops,
            "alu": self.alu,
            "sfu": self.sfu,
            "add": self.add,
            "sub": self.sub,
            "mul": self.mul,
            "div": self.div,
            "fma": self.fma,
            "pow": self.pow,
            "sqrt": self.sqrt,
            "exp": self.exp,
            "log": self.log,
            "trig": self.trig,
            "hyperbolic": self.hyperbolic,
            "logic": self.logic,
            "round": self.round,
            "sign": self.sign,
            "rcp": self.rcp,
        }
        return data

    @classmethod
    def from_dict(cls, d: dict) -> "ComputationInfo":
        obj = cls()
        # 只还原原始计数属性 (flops/alu/sfu 为派生 property, 不可赋值)
        for key, value in d.items():
            if key in obj.__dict__:
                setattr(obj, key, value)
        return obj


class MemoryAccessInfo:
    """访存统计类（功能阶段+访问模式）"""

    def __init__(self):
        self.read_weight = 0  # 读取权重/参数
        self.read_act = 0  # 读取输入张量
        self.write_act = 0  # 写入输出张量
        self.read_temp = 0  # 内部中间计算数据读取
        self.write_temp = 0  # 内部结果写入（缓存）

    # 计算访存次数（MAC），不包含临时读写
    @property
    def mac(self) -> int:
        return self.read_act + self.read_weight + self.write_act

    @property
    def mac_wtemp(self) -> int:
        return (
            self.read_act
            + self.read_weight
            + self.read_temp
            + self.write_act
            + self.write_temp
        )

    @property
    def read_act_weight(self) -> int:
        return self.read_act + self.read_weight

    @property
    def read_act_weight_temp(self) -> int:
        return self.read_act + self.read_weight + self.read_temp

    @property
    def write_act_temp(self) -> int:
        return self.write_act + self.write_temp

    def to_dict(self) -> dict:
        data = {
            "mac": self.mac,
            "mac_wtemp": self.mac_wtemp,
            "read_act": self.read_act,
            "read_weight": self.read_weight,
            "read_temp": self.read_temp,
            "read_act_weight": self.read_act_weight,
            "read_act_weight_temp": self.read_act_weight_temp,
            "write_act": self.write_act,
            "write_temp": self.write_temp,
            "write_act_temp": self.write_act_temp,
        }
        return data

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryAccessInfo":
        obj = cls()
        # 只还原原始计数属性 (mac/mac_wtemp 等为派生 property, 不可赋值)
        for key, value in d.items():
            if key in obj.__dict__:
                setattr(obj, key, value)
        return obj
