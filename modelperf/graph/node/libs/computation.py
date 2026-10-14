import math
from typing import Dict, List, Any

from ..operators import (
    ELEMENTWISE_OPS,
    REDUCTION_OPS,
    TENSOR_OPS,
    NORMALIZATION_OPS,
    ACTIVATION_OPS,
    MATRIX_OPS,
    CONV_OPS,
    RNN_OPS,
    ATTENTION_OPS,
)
from ..definition import ComputationInfo
from ...utils import numel, safe_axis
from .args import get_arg
from ....diagnostics import issue


class Computation:
    def __init__(
        self,
        op_type: str,
        op_args: Dict[str, Any],
        in_tensor_infos: List[Dict[str, Any]],
        out_tensor_infos: List[Dict[str, Any]],
        p_tensor_infos: List[Dict[str, Any]],
    ):
        # 记录算子类型和参数
        self.op_type = op_type
        self.op_args = op_args

        # 记录输入输出张量形状
        self.in_shapes = [tensor_info["shape"] for tensor_info in in_tensor_infos]
        self.p_shapes = [tensor_info["shape"] for tensor_info in p_tensor_infos]
        self.out_shapes = [tensor_info["shape"] for tensor_info in out_tensor_infos]

        # 记录第一个输入输出张量形状（非常常用）
        self.in0_shape = self.in_shapes[0] if len(self.in_shapes) > 0 else None
        self.out0_shape = self.out_shapes[0] if len(self.out_shapes) > 0 else None

        # 初始化结果对象
        self.ci = ComputationInfo()

    def _get_arg(self, key: str, default=None, required=False):
        return get_arg(
            self.op_args,
            key,
            default=default,
            required=required,
            op_type=self.op_type,
        )

    def elementwise(self, op_type, in_size=None, out_size=None):
        # in_shape[0] = in_shape[1] = in_shape[n-1] = out_shape[0]
        # in_size = n * out_size, n表示逐元素运算的元素数
        out_size = numel(self.out0_shape) if out_size is None else out_size

        if op_type == "Add":
            # Performs element-wise binary addition (with Numpy-style broadcasting support).
            self.ci.add += out_size

        elif op_type == "Sub":
            # Performs element-wise binary subtraction (with Numpy-style broadcasting support).
            self.ci.sub += out_size

        elif op_type == "Mul":
            # Performs element-wise binary multiplication (with Numpy-style broadcasting support).
            self.ci.mul += out_size

        elif op_type == "Div":
            # Performs element-wise binary division (with Numpy-style broadcasting support).
            self.ci.div += out_size

        elif op_type == "Pow":
            # Pow takes input data (Tensor) and exponent Tensor, and produces one output data (Tensor)
            # where the function f(x) = x^exponent, is applied to the data tensor elementwise.
            self.ci.pow += out_size

        elif op_type == "Sqrt":
            # Square root takes one input data (Tensor) and produces one output data (Tensor)
            # where the square root is, y = x^0.5, is applied to the tensor elementwise. If x is negative, then it will return NaN.
            self.ci.sqrt += out_size

        elif op_type == "Mod":
            # Performs an element-wise binary modulo operation.
            # The semantics and supported data types depend on the value of the fmod attribute which must be 0 (default), or 1.
            # if fmod == 0, then the operation is element-wise modulo, i.e. x - floor(x / y) * y.
            # if fmod == 1, then the operation is element-wise floating-point modulo, i.e. x - trunc(x / y) * y.
            self.ci.div += out_size  # 1. x / y
            self.ci.round += out_size  # 2. trunc/floor(x / y), 向零取整
            self.ci.mul += out_size  # 3. trunc/floor(x / y) * y
            self.ci.sub += out_size  # 4. x - trunc/floor(x / y) * y

        elif op_type in ["Neg", "Abs"]:
            # Neg takes one input data (Tensor) and produces one output data (Tensor) where each element flipped sign,
            # y = -x/abs(x), is applied to the tensor elementwise.
            self.ci.sign += out_size

        elif op_type == "Reciprocal":
            # Reciprocal takes one input data (Tensor) and produces one output data (Tensor) where the reciprocal is,
            # y = 1/x, is applied to the tensor elementwise.
            self.ci.rcp += out_size

        elif op_type in ["Ceil", "Floor", "Round"]:
            # Ceil takes one input data (Tensor) and produces one output data (Tensor) where the ceil is,
            # y = ceil(x), is applied to the tensor elementwise. If x is integral, +0, -0, NaN, or infinite, x itself is returned.
            # Floor takes one input data (Tensor) and produces one output data (Tensor) where the floor is,
            # y = floor(x), is applied to the tensor elementwise. If x is integral, +0, -0, NaN, or infinite, x itself is returned.
            # Round takes one input Tensor and rounds the values, element-wise, meaning it finds the nearest integer for each value.
            # In case of halves, the rule is to round them to the nearest even integer.
            # If input x is integral, +0, -0, NaN, or infinite, x itself is returned.
            # The output tensor has the same shape and type as the input.
            self.ci.round += out_size

        elif op_type in [
            "Equal",
            "Greater",
            "GreaterOrEqual",
            "Less",
            "LessOrEqual",
        ]:
            # Returns the tensor resulted from performing the equal/greater/greaterorequal/less/lessorequal
            # logical operation elementwise on the input tensors A and B (with Numpy-style broadcasting support).
            self.ci.logic += out_size

        elif op_type in [
            "And",
            "Or",
            "Xor",
            "Not",
            "BitwiseAnd",
            "BitwiseOr",
            "BitwiseXor",
            "BitwiseNot",
        ]:
            # Returns the tensor resulted from performing the and/or/xor/not/bitwiseand/bitwiseor/bitwisexor/bitwisenot
            # logical operation elementwise on the input tensors A and B (with Numpy-style broadcasting support).
            self.ci.logic += out_size

        elif op_type in ["Sin", "Cos", "Tan", "Asin", "Acos", "Atan"]:
            # Calculates the sine/cosine/tangent/arcsine/arccosine/arctangent of the given input tensor, element-wise.
            self.ci.trig += out_size

        elif op_type in ["Sinh", "Cosh", "Tanh", "Asinh", "Acosh", "Atanh"]:
            # Calculates the hyperbolic sine/cosine/tangent/arcsine/arctangent of the given input tensor, element-wise.
            self.ci.hyperbolic += out_size

        elif op_type == "Exp":
            # Calculates the exponential of the given input tensor, element-wise.
            self.ci.exp += out_size

        elif op_type == "Log":
            # Calculates the natural log of the given input tensor, element-wise.
            self.ci.log += out_size

        elif op_type == "Erf":
            # Computes the error function of the given input tensor, element-wise.
            # GELU 常被导出为 Erf 分解, 计一次超越函数开销。
            self.ci.exp += out_size

        elif op_type == "NonZero":
            # NonZero: Returns the indices of the elements that are non-zero.
            # 输出形状为 (rank, nnz)，与扫描量无关；代价应等于被扫描的输入元素数，
            # 不能按 out_size(rank*nnz) 计。
            self.ci.logic += numel(self.in0_shape)

        elif op_type in ["Where", "IsInf", "IsNaN"]:
            # Where: Return elements, either from X or Y, depending on condition.
            # IsInf: Map infinity to true and other values to false.
            # IsNaN: Returns which elements of the input are NaN.
            self.ci.logic += out_size

        elif op_type in ["Cast", "CastLike"]:
            # The operator casts the elements of a given input tensor to a data type specified by
            # the ‘to’ argument and returns an output tensor of the same size in the converted type.
            # The operator casts the elements of a given input tensor (the first input) to the same data type
            # as the elements of the second input tensor.
            # 类型转换: 逐元素拷贝, 计算代价可忽略, 不单独计 flops
            pass

        elif op_type == "Sign":
            # If input > 0, output 1.
            # if input < 0, output -1.
            # if input == 0, output 0.
            self.ci.logic += out_size * 2

        elif op_type == "Shrink":
            # If x < -lambd, y = x + bias;
            # If x > lambd, y = x - bias;
            # Otherwise, y = 0.
            self.ci.logic += out_size * 2

        elif op_type == "Clip":
            # Clip operator limits the given input within an interval.
            # Min(max, Max(input, min))
            self.ci.logic += out_size * 2

        else:
            issue(
                "unknown_subop",
                f"Unknown elementwise operation: {op_type}",
                op_type=op_type,
            )

    def reduction(self, op_type, in_size=None, out_size=None):
        # in_shape[0] != out_shape[0], in_size != out_size
        # 一般情况下， reduce操作一个主要输入和输出， 输入和输出形状和元素数不同
        in_size = numel(self.in0_shape) if in_size is None else in_size
        out_size = numel(self.out0_shape) if out_size is None else out_size
        reduction_factor = (
            in_size // out_size
        )  # 归约因子 = 输入元素数 / 规约后元素数(此处=输出元素数)
        per_output_ops = reduction_factor - 1  #  每个规约后元素需要的操作次数

        if op_type in ["ReduceMax", "ReduceMin"]:
            # Computes the max/min of the input tensor’s elements along the provided axes.
            """
            最大/最小值归约：
            计算步骤：
            1. 对每个输出位置，比较reduction_factor个输入元素
            操作：cmp = out_size * (reduction_factor - 1)
            中间变量：当前最大/最小值 (out_size)
            """
            self.ci.logic += out_size * per_output_ops

        elif op_type == "ReduceSum":
            # Computes the sum of the input tensor’s elements along the provided axes.
            """
            求和归约：
            计算步骤：
            1. 对每个输出位置，累加reduction_factor个输入元素
            操作：add = out_size * (reduction_factor - 1)
            中间变量：部分和 (out_size)
            """
            self.ci.add += out_size * per_output_ops

        elif op_type == "ReduceMean":
            # Computes the mean of the input tensor’s elements along the provided axes.
            """
            均值归约：
            计算步骤：
            1. 先求和（同ReduceSum）
            2. 然后除以元素数量
            操作：add + div
            中间变量：部分和 (out_size)
            """
            self.ci.add += out_size * per_output_ops  # sum(x)
            self.ci.mul += out_size  # NOTE, sum(x) * (1/n)

        elif op_type == "ReduceProd":
            # Computes the product of the input tensor’s elements along the provided axes.
            """
            乘积归约：
            计算步骤：
            1. 对每个输出位置，连乘reduction_factor个输入元素
            操作：mul = out_size * (reduction_factor - 1)
            中间变量：部分积 (out_size)
            """
            self.ci.mul += out_size * per_output_ops

        elif op_type == "ReduceL1":
            # Computes the L1 norm of the input tensor’s elements along the provided axes.
            """
            L1范数归约（绝对值和）：
            计算步骤：
            1. 对每个输入元素取绝对值
            2. 求和归约
            操作：abs + add
            中间变量：仅需存储部分和 (out_size)
            """
            self.ci.sign += in_size  # abs
            self.ci.add += out_size * per_output_ops  # reduce sum

        elif op_type == "ReduceL2":
            # Computes the L2 norm of the input tensor’s elements along the provided axes.
            """
            L2范数归约（平方和开方）：
            步骤：
            1. 平方 (x^2)
            2. 求和归约
            3. 开方
            操作：mul + add + sqrt
            中间变量：部分和 (out_size)
            """
            self.ci.mul += in_size  # x^2
            self.ci.add += out_size * per_output_ops  # reduce sum
            self.ci.sqrt += out_size  # sqrt

        elif op_type == "ReduceLogSum":
            # Computes the log sum of the input tensor’s elements along the provided axes.
            """
            求和后取对数：
            操作：add + log
            中间变量：部分和 (out_size)
            """
            self.ci.add += out_size * per_output_ops  # reduce sum
            self.ci.log += out_size  # log

        elif op_type == "ReduceLogSumExp":
            # Computes the log sum exponent of the input tensor’s elements along the provided axes.
            """
            指数求和取对数：
            操作：exp + add + log
            中间变量：exp(x) (in_size) + 部分和 (out_size)
            """
            self.ci.exp += in_size  # exp(x)
            self.ci.add += out_size * per_output_ops  # reduce sum
            self.ci.log += out_size  # log

        elif op_type == "ReduceSumSquare":
            # Computes the sum square of the input tensor’s elements along the provided axes.
            """
            平方求和：
            操作：mul + add
            中间变量：部分和 (out_size)
            """
            self.ci.mul += in_size  # x^2
            self.ci.add += out_size * per_output_ops  # reduce sum

        elif op_type in ["ArgMax", "ArgMin"]:
            # Computes the indices of the max/min elements of the input tensor’s elements along the provided axis.
            """
            极值索引归约：
            计算步骤：
            1. 遍历所有输入元素，记录当前极值的索引
            操作：对每个输出位置，比较reduction_factor个输入元素
            注意：无需中间存储，直接更新索引
            """
            self.ci.logic += out_size * per_output_ops

        elif op_type == "TopK":
            # Retrieve the top-K largest or smallest elements along a specified axis.
            # axis - INT (default is '-1')
            axis = safe_axis(self._get_arg("axis", -1), len(self.in0_shape))
            N = self.in0_shape[axis]
            K = self.out0_shape[axis]  # Outputs: Values and Indices
            other_size = numel(
                [dim for i, dim in enumerate(self.in0_shape) if i != axis]
            )
            self.ci.logic += other_size * (K + (N - K) * math.log2(N))  # 堆选择法

        elif op_type == "Unique":
            # Find the unique elements of a tensor.
            # input: X
            # output: Y, indices, inverse_indices, counts
            # axis: (Optional) The dimension to apply unique. If not specified, the unique elements of the flattened input are returned.
            axis = safe_axis(self._get_arg("axis", None), len(self.in0_shape))

            # 排序法
            if axis is None:
                L = in_size
                M = 1
            else:
                L = self.in0_shape[axis]
                M = in_size // L
            self.ci.logic += M * L * math.log2(L)

            # # 哈希法
            # if axis is None:
            #     self.ci.logic += in_size
            # else:
            #     L = self.in_shapes[0][axis]
            #     M = in_size // L
            #     self.ci.logic += M * L  # = in_size

        elif op_type == "AveragePool":
            """
            AveragePool（窗口平均）：
            计算步骤（每个输出位置）：
            1. 窗口内求和 (ReduceSum)
            2. 乘以 1/窗口元素个数（等价于除法）

            操作：add + mul
            中间变量：
                - sum(window) (out_size)
            复杂度（按元素计数）：
                - 加法：每个输出位置 per_output_ops = kernel_elems - 1
                - 乘法：每个输出位置 1 次（乘以 1/kernel_elems）
            """
            # kernel_shape - INTS (required)
            kernel_shape = self._get_arg("kernel_shape", required=True)
            per_output_ops = max(numel(kernel_shape) - 1, 0)

            self.ci.add += out_size * per_output_ops  # ReduceSum 累加
            self.ci.mul += out_size  # NOTE, 乘以1/kernel_elems（等价除法）

        elif op_type == "MaxPool":
            """
            MaxPool（窗口最大值）：
            计算步骤（每个输出位置）：
            1. 窗口内比较取最大值 (ReduceMax)

            操作：cmp
            中间变量：
                - max(window) (out_size)
            复杂度（按元素计数）：
                - 比较：每个输出位置 per_output_ops = kernel_elems - 1
            """
            # kernel_shape - INTS (required)
            kernel_shape = self._get_arg("kernel_shape", required=True)
            per_output_ops = max(numel(kernel_shape) - 1, 0)

            self.ci.logic += out_size * per_output_ops  # ReduceMax 比较操作

        elif op_type == "GlobalAveragePool":
            """
            GlobalAveragePool（全局平均）：
            计算步骤（每个输出通道）：
            1. 对该通道所有空间元素求和 (ReduceSum)
            2. 乘以 1/N（N 为该通道的空间元素个数）

            操作：add + mul
            中间变量：
                - sum(channel) (out_size)
            复杂度（按元素计数）：
                - 加法：每个输出需 (N - 1) 次
                - 乘法：每个输出 1 次（乘以 1/N）
            """
            self.ci.add += out_size * per_output_ops  # ReduceSum 累加
            self.ci.mul += out_size  # NOTE, 乘以1/N（等价除法）

        elif op_type == "GlobalMaxPool":
            """
            GlobalMaxPool（全局最大）：
            计算步骤（每个输出通道）：
            1. 对该通道所有空间元素取最大值 (ReduceMax)

            操作：cmp
            中间变量：
                - max(channel) (out_size)
            复杂度（按元素计数）：
                - 比较：每个输出需 (N - 1) 次
            """
            self.ci.logic += out_size * per_output_ops  # ReduceMax 比较操作

        else:
            issue(
                "unknown_subop",
                f"Unknown reduction operation: {op_type}",
                op_type=op_type,
            )

    def tensor(self, op_type, in_size=None, out_size=None):
        # out_size = (
        #     sum(numel(out_shape) for out_shape in self.out_shapes)
        #     if out_size is None
        #     else out_size
        # )  # TODO, 输出一定都是有效输出吗？
        out_size = numel(self.out0_shape) if out_size is None else out_size

        if op_type in [
            "EyeLike",  # Generate a 2D tensor (matrix) with ones on the diagonal and zeros everywhere else.
            "OneHot",  # Produces a one-hot tensor based on inputs.
            "Range",  # Generate a tensor containing a sequence of numbers that begin at start and extends by increments of delta up to limit (exclusive).
            "RandomNormal",  # Generate a tensor with random values drawn from a normal distribution.
            "RandomNormalLike",  # Generate a tensor with random values drawn from a normal distribution.
            "RandomUniform",  # Generate a tensor with random values drawn from a uniform distribution.
            "RandomUniformLike",  # Generate a tensor with random values drawn from a uniform distribution.
        ]:
            if op_type == "EyeLike":
                self.ci.logic += out_size  # 对每个输出元素判断是否在对角线（i==j）
            elif op_type == "OneHot":
                self.ci.logic += out_size  # 对每个输出元素判断是否为索引值
            elif op_type == "Range":
                self.ci.add += out_size  # 每个元素 = 前一个元素 + step
            elif op_type in [
                "RandomNormal",
                "RandomNormalLike",
                "RandomUniform",
                "RandomUniformLike",
            ]:
                # 随机数生成: 代价取决于 RNG 实现, 暂不建模
                pass
        elif op_type == "Resize":
            """
            ONNX Resize（已知输出形状版）
            口径：
            - 最近邻：视作 0 flops（仅寻址）
            - 线性/三次：每输出做 T 次“值×权重”的加权累加
                * FMA：fma += |Y| * T
                * mul/add：mul += |Y| * T；add += |Y| * (T-1)
            - I/O 上界（无复用）：读 |Y|*T，写 |Y|
            仅需：self.in_shapes[0]，self.out_shapes[0]（或 op_args['out_shape']）
            """
            # =====(0) 形状/参数=====
            # mode - STRING (default is 'nearest')
            mode = self._get_arg("mode", "nearest")  # "nearest"|"linear"|"cubic"
            if mode == "nearest":
                return

            X = self.in0_shape  # 输入形状 (N-D)
            Y = self.out0_shape  # 输出形状 (N-D)
            N = len(X)
            if len(Y) != N:
                issue(
                    "op_shape",
                    f"Resize: out_shape rank mismatch (in rank={N}, out rank={len(Y)})",
                    op_type=op_type,
                )
                return

            # antialias - INT (default is '0')
            antialias = self._get_arg("antialias", 0)  # 仅 linear/cubic 下采生效
            # axes - INTS
            # If provided, it specifies a subset of axes that ‘roi’, ‘scales’ and ‘sizes’ refer to.
            # If not provided, all axes are assumed [0, 1, …, r-1], where r = rank(data).
            axes = self._get_arg("axes", list(range(N)))  # 未给则全轴可缩放
            axes = [safe_axis(axis, N) for axis in axes]
            if len(set(axes)) != len(axes):
                issue(
                    "op_shape",
                    "Resize: axes contains duplicates",
                    op_type=op_type,
                    axes=axes,
                )
                return

            # =====(1) 每轴缩放比 s_i（由形状直接得到）=====
            # 非可缩放轴视为 s_i = 1.0（即不改变大小，不参与滤波 taps）
            s = [(Y[i] / X[i]) if i in axes else 1.0 for i in range(N)]

            # =====(2) 每轴 taps（antialias：下采时扩大支持域）=====
            base = {"linear": 2, "cubic": 4}.get(mode, 1)
            taps_per_axis = []
            for i in range(N):
                if i not in axes:
                    taps_per_axis.append(1)
                    continue
                si = max(float(s[i]), 1e-12)
                if antialias and mode in ("linear", "cubic") and si < 1.0:
                    # 规范：滤波器按 max(1, 1/scale) 拉伸；下采时 taps 放大
                    taps_per_axis.append(math.ceil(base * (1.0 / si)))
                else:
                    taps_per_axis.append(base)

            # =====(3) flops（FMA 口径；最近邻≈0）=====
            self.ci.fma += numel(Y) * numel(taps_per_axis)
            # 若偏好 mul/add 口径：
            # self.ci.mul += numel(Y) * numel(taps_per_axis)
            # self.ci.add += numel(Y) * (numel(taps_per_axis) - 1)

        elif op_type == "Upsample":
            """
            ONNX Upsample（已知输出形状版；仅上采样，nearest/linear，无 antialias）
            口径同 Resize(antialias=0)：
            - nearest：≈0 flops
            - linear：fma += |Y| * T，其中 T = ∏(每轴 tap=2)
            仅需：self.in_shapes[0]，self.out_shapes[0]（或 op_args['out_shape')）
            """
            # =====(0) 形状/参数=====
            # mode - STRING (default is 'nearest')
            mode = self._get_arg("mode", "nearest")  # "nearest"|"linear"
            if mode == "nearest":
                return

            X = self.in0_shape  # 输入形状 (N-D)
            Y = self.out0_shape  # 输出形状 (N-D)
            N = len(X)
            if len(Y) != N:
                issue(
                    "op_shape",
                    f"Upsample: out_shape rank mismatch (in rank={N}, out rank={len(Y)})",
                    op_type=op_type,
                )
                return

            # ONNX Upsample 没有 axes；按规范用 scales（>=1），
            # 这里用 Y 与 X 的比值作为实际 scale 来判断是否为上采样。
            # 仅上采样约束（按形状校验）
            for i in range(N):
                if Y[i] < X[i]:
                    issue(
                        "op_shape",
                        "Upsample requires scales >= 1.0 (per axis)",
                        op_type=op_type,
                        axis=i,
                    )
                    return

            # 线性插值：每个被放大维的 tap=2；未放大（scale==1）的维度 tap=1
            T = 1
            for i in range(N):
                tap = 2 if Y[i] > X[i] else 1
                T *= tap

            self.ci.fma += numel(Y) * T
            # mul/add 口径：
            # self.ci.mul += numel(Y) * T
            # self.ci.add += numel(Y) * (T - 1)

        elif op_type in [
            "Reshape",
            "Squeeze",
            "Unsqueeze",
            "Flatten",
            "Expand",
            "Shape",
            "Size",
            "Identity",
            "Transpose",
            "DepthToSpace",
            "SpaceToDepth",
            "ReverseSequence",
            "Slice",
            "Split",
            "Concat",
            "Tile",
            "Pad",
            "Gather",
            "GatherElements",
            "GatherND",
            "Scatter",
            "ScatterElements",
            "ScatterND",
            "Constant",
            "ConstantOfShape",
        ]:
            pass

        else:
            issue(
                "unknown_subop",
                f"Unknown tensor operation: {op_type}",
                op_type=op_type,
            )

    def normalization(self, op_type, in_size=None, out_size=None):
        # in_shape[0] = out_shape[0], in_size = out_size
        # 归一化操作一般都是一个主要输入和输出， 输入和输出形状和元素数相同
        in_size = numel(self.in0_shape) if in_size is None else in_size
        if op_type == "BatchNormalization":
            # training_mode - INT(default is "0")
            training_mode = self._get_arg("training_mode", 0)
            # NOTE, [C], 规约后的形状, 除C外的维度都进行规约
            reduced_size = self.in0_shape[1]
            if training_mode:
                reduction_factor = in_size // reduced_size
                per_output_ops = reduction_factor - 1
                # cur_mean = ReduceMean(X, axis=all_except_channel_index)
                self.ci.add += reduced_size * per_output_ops  # sum(x)
                self.ci.mul += reduced_size  # NOTE, sum(x) / reduced_size
                # cur_var = ReduceVar(X, axis=all_except_channel_index)
                self.ci.sub += in_size  # x - mean
                self.ci.mul += in_size  # (x - mean)^2
                self.ci.add += reduced_size * per_output_ops  # sum((x - mean)^2)
                self.ci.mul += reduced_size  # NOTE, sum((x - mean)^2) / reduced_size

                # mean = mean * mom + cur_mean * (1 - mom)
                self.ci.mul += reduced_size  # mean * mom
                self.ci.mul += (
                    reduced_size  # cur_mean * (1 - mom), NOTE, 1和mom都是常数
                )
                self.ci.add += reduced_size  # mean * mom + cur_mean * (1 - mom)
                # var = var * mom + cur_var * (1 - mom)
                self.ci.mul += reduced_size  # var * mom
                self.ci.mul += reduced_size  # cur_var * (1 - mom), NOTE, 1和mom都是常数
                self.ci.add += reduced_size  # var * mom + cur_var * (1 - mom)

            # gamma * [(x - mean) / sqrt(var + eps)] + beta
            self.ci.sub += in_size  # 1. x - mean
            self.ci.add += reduced_size  # 2. var + eps
            self.ci.sqrt += reduced_size  # 3. std = sqrt(var + eps)
            self.ci.div += in_size  # 4. (x-mean)/std
            self.ci.mul += in_size  # 5. gamma * normalized
            self.ci.add += in_size  # 6. + beta

        elif op_type == "LayerNormalization":
            # 对X[axis:]进行规约，规约后的形状为X[:axis]
            # axis - INT (default is '-1')
            axis = safe_axis(self._get_arg("axis", -1), len(self.in0_shape))
            reduced_size = numel(self.in0_shape[:axis])
            reduction_factor = in_size // reduced_size
            per_output_ops = reduction_factor - 1
            # cur_mean = ReduceMean(X, axis=all_except_channel_index)
            self.ci.add += reduced_size * per_output_ops  # sum(x)
            self.ci.mul += reduced_size  # NOTE, sum(x) / reduced_size
            # cur_var = ReduceVar(X, axis=all_except_channel_index)
            self.ci.sub += in_size  # x - mean
            self.ci.mul += in_size  # (x - mean)^2
            self.ci.add += reduced_size * per_output_ops  # sum((x - mean)^2)
            self.ci.mul += reduced_size  # NOTE, sum((x - mean)^2) / reduced_size

            # gamma * [(x - mean) / sqrt(var + eps)] + beta
            self.ci.sub += in_size  # 1. x - mean
            self.ci.add += reduced_size  # 2. var + eps
            self.ci.sqrt += reduced_size  # 3. std = sqrt(var + eps)
            self.ci.div += in_size  # 4. (x-mean)/std
            self.ci.mul += in_size  # 5. gamma * normalized
            self.ci.add += in_size  # 6. + beta

        elif op_type == "InstanceNormalization":
            # 对空间维度进行规约，规约后的形状为[N, C]
            reduced_size = self.in0_shape[0] * self.in0_shape[1]
            reduction_factor = in_size // reduced_size
            per_output_ops = reduction_factor - 1
            # cur_mean = ReduceMean(X, axis=all_except_channel_index)
            self.ci.add += reduced_size * per_output_ops  # sum(x)
            self.ci.mul += reduced_size  # NOTE, sum(x) / reduced_size
            # cur_var = ReduceVar(X, axis=all_except_channel_index)
            self.ci.sub += in_size  # x - mean
            self.ci.mul += in_size  # (x - mean)^2
            self.ci.add += reduced_size * per_output_ops  # sum((x - mean)^2)
            self.ci.mul += reduced_size  # NOTE, sum((x - mean)^2) / reduced_size

            # gamma * [(x - mean) / sqrt(var + eps)] + beta
            self.ci.sub += in_size  # 1. x - mean
            self.ci.add += reduced_size  # 2. var + eps
            self.ci.sqrt += reduced_size  # 3. std = sqrt(var + eps)
            self.ci.div += in_size  # 4. (x - mean) / std
            self.ci.mul += in_size  # 5. gamma * normalized
            self.ci.add += in_size  # 6. + beta

        elif op_type == "GroupNormalization":
            # 对空间维度进行规约，规约后的形状为[N, G]
            # num_groups - INT(required)
            num_groups = self._get_arg("num_groups", required=True)
            reduced_size = self.in0_shape[0] * num_groups
            reduction_factor = in_size // reduced_size
            per_output_ops = reduction_factor - 1
            # cur_mean = ReduceMean(X, axis=all_except_channel_index)
            self.ci.add += reduced_size * per_output_ops  # sum(x)
            self.ci.mul += reduced_size  # NOTE, sum(x) / reduced_size
            # cur_var = ReduceVar(X, axis=all_except_channel_index)
            self.ci.sub += in_size  # x - mean
            self.ci.mul += in_size  # (x - mean)^2
            self.ci.add += reduced_size * per_output_ops  # sum((x - mean)^2)
            self.ci.mul += reduced_size  # NOTE, sum((x - mean)^2) / reduced_size

            # gamma * [(x - mean) / sqrt(var + eps)] + beta
            self.ci.sub += in_size  # 1. x - mean
            self.ci.add += reduced_size  # 2. var + eps
            self.ci.sqrt += reduced_size  # 3. std = sqrt(var + eps)
            self.ci.div += in_size  # 4. (x-mean)/std
            self.ci.mul += in_size  # 5. gamma * normalized
            self.ci.add += in_size  # 6. + beta

        elif op_type == "RMSNormalization":
            # 对X[axis:]进行规约，规约后的形状为X[:axis]
            # axis - INT (default is '-1')
            axis = safe_axis(self._get_arg("axis", -1), len(self.in0_shape))
            reduced_size = numel(self.in0_shape[:axis])
            reduction_factor = in_size // reduced_size
            per_output_ops = reduction_factor - 1
            # RMS = sqrt(mean(x^2) + eps)
            self.ci.mul += in_size  # x^2
            self.ci.add += reduced_size * per_output_ops  # sum(x)
            self.ci.mul += reduced_size  # NOTE, sum(x) / reduced_size
            self.ci.add += reduced_size  # mean(x^2) + eps
            self.ci.sqrt += reduced_size  # sqrt(mean(x^2) + eps)

            # gamma * [x / RMS]
            self.ci.div += in_size  # 1. x / (RMS + eps)
            self.ci.mul += in_size  # 2. gamma * normalized

        elif op_type == "LpNormalization":
            # Y = X / [(Σ|X|^p)^(1/p) + eps]
            # 对X[axis:]进行规约，规约后的形状为X[:axis]
            # axis - INT (default is '-1')
            axis = safe_axis(self._get_arg("axis", -1), len(self.in0_shape))
            reduced_size = numel(self.in0_shape[:axis])
            reduction_factor = in_size // reduced_size
            per_output_ops = reduction_factor - 1

            # p - INT (default is '2')
            p = self._get_arg("p", 2)
            # 1. 计算范数
            if p == 1:
                self.ci.sign += in_size  # abs(x)
                self.ci.add += reduced_size * per_output_ops  # sum
            elif p == 2:
                self.ci.mul += in_size  # x^2
                self.ci.add += reduced_size * per_output_ops  # sum
                self.ci.sqrt += reduced_size  # sqrt
            else:
                self.ci.pow += in_size  # x^p
                self.ci.add += reduced_size * per_output_ops  # sum
                self.ci.pow += reduced_size  # (sum)^(1/p)
            # 2. + eps
            self.ci.add += reduced_size
            # 3. x / norm
            self.ci.div += in_size

        elif op_type == "MeanVarianceNormalization":
            # axes - INTS (default is ['0', '2', '3'])
            axes = self._get_arg("axes", [0, 2, 3])
            axes = [safe_axis(a, len(self.in0_shape)) for a in axes]
            # 对指定轴进行规约，规约后的形状为[dim!=axes]
            reduced_size = numel(
                [
                    dim
                    for i, dim in enumerate(self.in0_shape)
                    if i not in axes  # not in axes
                ]
            )
            reduction_factor = in_size // reduced_size
            per_output_ops = reduction_factor - 1
            # cur_mean = ReduceMean(X, axis=all_except_channel_index)
            self.ci.add += reduced_size * per_output_ops  # sum(x)
            self.ci.mul += reduced_size  # NOTE, sum(x) / reduced_size
            # cur_var = ReduceVar(X, axis=all_except频道index)
            self.ci.sub += in_size  # x - mean
            self.ci.mul += in_size  # (x - mean)^2
            self.ci.add += reduced_size * per_output_ops  # sum((x - mean)^2)
            self.ci.mul += reduced_size  # NOTE, sum((x - mean)^2) / reduced_size

            # y = (x - mean) / sqrt(var + eps)
            self.ci.sub += in_size  # x - mean
            self.ci.add += reduced_size  # var + eps
            self.ci.sqrt += reduced_size  # sqrt(var + eps)
            self.ci.div += in_size  # (x - mean) / sqrt(var + eps)

        else:
            issue(
                "unknown_subop",
                f"Unknown normalization operation: {op_type}",
                op_type=op_type,
            )

    def activation(self, op_type, in_size=None, out_size=None):
        # in_shape[0] = out_shape[0], in_size = out_size
        # 归一化操作一般都是一个主要输入和输出， 输入和输出形状和元素数相同
        out_size = numel(self.out0_shape) if out_size is None else out_size

        if op_type == "Relu":
            # y = max(0, x)
            self.ci.logic += out_size

        elif op_type == "ThresholdedRelu":
            # y = x for x > alpha, y = 0 otherwise
            self.ci.logic += out_size

        elif op_type == "LeakyRelu":
            # f(x) = alpha * x for x < 0,
            # f(x) = x for x >= 0
            self.ci.logic += out_size
            self.ci.mul += out_size

        elif op_type == "PRelu":
            # f(x) = slope * x for x < 0,
            # f(x) = x for x >= 0
            self.ci.logic += out_size
            self.ci.mul += out_size

        elif op_type == "Sigmoid":
            # y = 1 / (1 + exp(-x))
            self.ci.sign += out_size  # 1. -x
            self.ci.exp += out_size  # 2. exp(-x)
            self.ci.add += out_size  # 3. 1 + exp(-x)
            self.ci.rcp += out_size  # 4. 1 / (1 + exp(-x))

        elif op_type == "Swish":
            # x * sigmoid(x), x * (1 / (1 + exp(-x))),
            self.ci.sign += out_size  # 1. -x
            self.ci.exp += out_size  # 2. exp(-x)
            self.ci.add += out_size  # 3. 1 + exp(-x)
            self.ci.rcp += out_size  # 4. 1 / (1 + exp(-x))
            self.ci.mul += out_size  # 5. x * sigmoid(x)

        elif op_type == "HardSigmoid":
            # y = max(0, min(1, alpha * x + beta))
            self.ci.mul += out_size  # 1. alpha * x
            self.ci.add += out_size  # 2. alpha * x + beta
            self.ci.logic += out_size * 2  # 3. min, max

        elif op_type == "HardSwish":
            # y = x * max(0, min(1, alpha * x + beta)) = x * HardSigmoid<alpha, beta>(x), where alpha = 1/6 and beta = 0.5
            # x * Relu6(x + 3) / 6, x * max(0, min(6, (x + 3))) / 6
            self.ci.add += out_size  # 1. x + 3
            self.ci.logic += out_size * 2  # 2. min, max
            self.ci.mul += out_size  # 3. NOTE, step2 / 6, step * (1/6)
            self.ci.mul += out_size  # 4. x * step3

        elif op_type == "Softplus":
            # # y = ln(exp(x) + 1)
            # self.ci.exp += out_size  # 1. exp(x)
            # self.ci.add += out_size  # 2. 1 + exp(x)
            # self.ci.log += out_size  # 3. log(1 + exp(x))

            # 数值稳定实现: y = max(x, 0) + log(1 + exp(−∣x∣))
            self.ci.logic += out_size  # 1. max(x, 0)
            self.ci.sign += out_size * 2  # 2. −∣x∣
            self.ci.exp += out_size  # 3. exp(−∣x∣)
            self.ci.add += out_size  # 4. 1 + exp(−∣x∣)
            self.ci.log += out_size  # 5. log(1 + exp(−∣x∣))
            self.ci.add += out_size  # 6. step3 + step5

        elif op_type == "Softsign":
            # x / (1 + |x|)
            self.ci.sign += out_size  # 1. |x|
            self.ci.add += out_size  # 2. 1 + |x|
            self.ci.div += out_size  # 3. x / (1 + |x|)

        elif op_type == "Elu":
            # f(x) = alpha * (exp(x) - 1.) for x < 0,
            # f(x) = x for x >= 0
            self.ci.logic += out_size  # 1. x >= 0 ?
            self.ci.exp += out_size  # 2. exp(x)
            self.ci.sub += out_size  # 3. exp(x) - 1
            self.ci.mul += out_size  # 4. alpha * (exp(x) - 1)

        elif op_type == "Selu":
            # y = gamma * (alpha * e^x - alpha) for x <= 0,
            # y = gamma * x for x > 0
            # scale * Elu(x), scale * (x >= 0 ? x : alpha * (exp(x) - 1)),
            self.ci.logic += out_size  # 1. x >= 0 ?
            self.ci.exp += out_size  # 2. exp(x)
            self.ci.sub += out_size  # 3. exp(x) - 1
            self.ci.mul += out_size  # 4. alpha * (exp(x) - 1)
            self.ci.mul += out_size  # 5. scale * step4

        elif op_type == "Gelu":
            """
            GELU近似公式: 0.5 * x * (1 + tanh(sqrt(2/π)*(x + 0.044715*x^3)))
            计算步骤分解:
            1. x^2 = x * x          → 1次mul
            2. x^3 = x^2 * x        → 1次mul
            3. 0.044715 * x^3      → 1次mul
            4. x + 0.044715*x^3    → 1次add
            5. sqrt(2/π) * step4   → 1次mul
            6. tanh(step5)         → 1次hyperbolic
            7. 1 + tanh            → 1次add
            8. x * 0.5             → 1次mul
            9. step8 * step7       → 1次mul
            总计: 6次mul + 2次add + 1次hyperbolic
            中间变量: x^3 (out_size), tanh结果 (out_size)
            """
            self.ci.mul += out_size * 6  # 步骤1/2/3/5/8/9
            self.ci.add += out_size * 2  # 步骤4/7
            self.ci.hyperbolic += out_size * 1  # 步骤6

        elif op_type == "Softmax":
            # Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1)
            """
            Softmax（数值稳定实现）：
            计算步骤：
            1. 计算每个归约维度的最大值 (ReduceMax)
            2. 输入减去最大值 (Sub)
            3. 取指数 (Exp)
            4. 求和归约 (ReduceSum)
            5. 归一化除法 (Div)
            操作：cmp + sub + exp + add + div
            """
            in_size = numel(self.in0_shape) if in_size is None else in_size
            # axis - INT (default is '-1')
            axis = safe_axis(self._get_arg("axis", -1), len(self.in0_shape))
            reduced_size = numel(
                [dim for i, dim in enumerate(self.in0_shape) if i != axis]
            )
            reduction_factor = (
                in_size // reduced_size
            )  # 归约因子 = 输入元素数 / 规约后元素数
            per_output_ops = reduction_factor - 1  #  每个规约后元素需要的操作次数

            self.ci.logic += reduced_size * per_output_ops  # ReduceMax
            self.ci.sub += in_size  # x - max
            self.ci.exp += in_size  # exp(x - max)
            self.ci.add += reduced_size * per_output_ops  # sum(exp(x - max))
            self.ci.div += in_size  # exp(x - max) / sum(exp(x - max))

        elif op_type == "LogSoftmax":
            # LogSoftmax(input, axis) = Log(Softmax(input, axis=axis))
            """
            LogSoftmax（数值稳定实现）：
            计算步骤：
            1. 计算每个归约维度的最大值 (ReduceMax)
            2. 输入减去最大值 (Sub)
            3. 取指数 (Exp)
            4. 求和归约 (ReduceSum)
            5. 对和取对数 (Log)
            6. 加回最大值 (Add)
            7. x - logsumexp (Sub)
            操作：cmp + sub + exp + add + log + add + sub
            中间变量：
                - max(x) (out_size)
                - exp(x - max) (in_size)
                - sum(exp(x - max)) (out_size)
                - logsumexp (out_size)
            """
            # axis - INT (default is '-1')
            axis = safe_axis(self._get_arg("axis", -1), len(self.in0_shape))
            reduced_size = numel(
                [dim for i, dim in enumerate(self.in0_shape) if i != axis]
            )
            reduction_factor = (
                in_size // reduced_size
            )  # 归约因子 = 输入元素数 / 规约后元素数
            per_output_ops = reduction_factor - 1  #  每个规约后元素需要的操作次数

            self.ci.logic += reduced_size * per_output_ops  # ReduceMax 比较操作
            self.ci.sub += in_size  # x - max
            self.ci.exp += in_size  # exp(x - max)
            self.ci.add += reduced_size * per_output_ops  # sum(exp(x - max))
            self.ci.log += reduced_size  # log(sum(exp(x - max)))
            self.ci.sub += in_size  # x - max - log(sum(exp(x - max)))

        else:
            issue(
                "unknown_subop",
                f"Unknown activation operation: {op_type}",
                op_type=op_type,
            )

    def matrix(self, op_type, in_size=None, out_size=None):
        in_shapes = self.in_shapes + self.p_shapes  # NOTE
        if op_type == "MatMul":
            """
            MatMul（矩阵乘法）:
            计算公式：
                Y = A @ B
            其中：
                - A 为形状为 (..., M, K1) 的张量
                - B 为形状为 (..., K2, N) 的张量
                - Y 为形状为 (..., M, N) 的张量
            """
            shape_a, shape_b = in_shapes[0], in_shapes[1]
            if len(shape_a) < 2 or len(shape_b) < 2:
                issue(
                    "op_shape",
                    f"MatMul: 输入秩不足 2 (a rank={len(shape_a)}, b rank={len(shape_b)})",
                    op_type=op_type,
                )
                return
            M, K1 = shape_a[-2], shape_a[-1]
            K2, N = shape_b[-2], shape_b[-1]
            if K1 != K2:
                issue(
                    "op_shape",
                    f"MatMul: 内维不匹配 (K1={K1}, K2={K2})",
                    op_type=op_type,
                )
                return

            # 计算广播后的批维
            ba = shape_a[:-2]
            bb = shape_b[:-2]
            # 右对齐广播
            max_len = max(len(ba), len(bb))
            ba = [1] * (max_len - len(ba)) + ba
            bb = [1] * (max_len - len(bb)) + bb
            batch = []
            for xa, xb in zip(ba, bb):
                if xa == xb or xa == 1 or xb == 1:
                    batch.append(max(xa, xb))
                else:
                    issue(
                        "op_shape",
                        f"MatMul: 批维不可广播 (a={ba}, b={bb})",
                        op_type=op_type,
                    )
                    return
            B = numel(batch) if batch else 1

            # 每个输出元素Y[..., i, j]是长度为K的点积
            self.ci.fma += B * M * N * K1

        elif op_type == "Gemm":
            """
            Gemm（General Matrix Multiplication）:
            计算公式：
                Y = alpha * (A' @ B') + beta * C
            其中：
                - A' = transpose(A) if transA
                - B' = transpose(B) if transB
                - alpha, beta 为缩放系数
            """

            # === 解析输入 ===
            # onnx Gemm 最多 3 个输入：A, B, C
            if len(in_shapes) < 2:
                issue(
                    "op_shape",
                    f"Gemm: 至少需要 2 个输入 (A,B), 实际 {len(in_shapes)}",
                    op_type=op_type,
                )
                return

            A_shape = in_shapes[0]
            B_shape = in_shapes[1]
            C_shape = in_shapes[2] if len(in_shapes) > 2 else None

            # === 读取属性 ===
            alpha = self._get_arg("alpha", 1.0)  # alpha - FLOAT (default is '1.0')
            beta = self._get_arg("beta", 1.0)  # beta - FLOAT (default is '1.0')
            transA = bool(self._get_arg("transA", 0))  # transA - INT (default is '0')
            transB = bool(self._get_arg("transB", 0))  # transB - INT (default is '0')

            # === 确定有效形状 ===
            A_m, A_k = (A_shape[1], A_shape[0]) if transA else (A_shape[0], A_shape[1])
            B_k, B_n = (B_shape[1], B_shape[0]) if transB else (B_shape[0], B_shape[1])
            if A_k != B_k:
                issue(
                    "op_shape",
                    f"Gemm: 内维不兼容 (A_k={A_k}, B_k={B_k})",
                    op_type=op_type,
                )
                return

            M, N, K = A_m, B_n, A_k

            # === 广播处理（仅对 C）===
            # C 必须能广播成 (M, N)
            if C_shape:
                # 简单验证兼容性（不展开广播逻辑）
                if len(C_shape) > 2:
                    issue(
                        "op_shape",
                        f"Gemm: C 形状 {C_shape} 非法 (秩应 <= 2)",
                        op_type=op_type,
                    )
                    return
                c_m = C_shape[-2] if len(C_shape) >= 2 else 1
                c_n = C_shape[-1] if len(C_shape) >= 1 else 1
                if not ((c_m in (1, M)) and (c_n in (1, N))):
                    issue(
                        "op_shape",
                        f"Gemm: C 形状 {C_shape} 无法广播到 ({M}, {N})",
                        op_type=op_type,
                    )
                    return

            # === 批维（Gemm 是单矩阵乘，不带batch） ===
            B = 1

            # === 主矩阵乘法计算 ===
            self.ci.fma += B * M * N * K  # 主乘加次数

            # === α 缩放部分 ===
            if alpha != 1.0:
                self.ci.mul += B * M * N  # alpha * (A@B)

            # === β*C 部分 ===
            if beta != 0.0 and C_shape is not None:
                self.ci.mul += B * M * N  # beta * C
                self.ci.add += B * M * N  # (alpha*A@B) + (beta*C)

        else:
            issue(
                "unknown_subop",
                f"Unknown matrix operation: {op_type}",
                op_type=op_type,
            )

    def conv(self, op_type, in_size=None, out_size=None):
        """
        Conv（卷积）:
        统计目标：乘加次数（FMA）与偏置加法次数（add）

        形状约定（通用 NCHW / NCDHW 等，d 维）：
            - X: (N, C_in, *S_out对应的空间前的输入空间) —— 本处只用到 N、C_in
            - W: Conv (C_out, C_in/groups, *K) / ConvTranspose (C_in, C_out/groups, *K)               —— 用到 C_out、核尺寸 K
            - Y: (N, C_out, *S_out)                     —— 用到 C_out、输出空间尺寸 S_out
        注：一旦 Y 已知，stride/pad/dilation 只影响 Y 的大小；**flops 完全由 (C_in, C_out, K, groups, Y) 决定**。
        注：ConvTranspose 权重为 (C_in, C_out/groups, *K)，以“输入元素”为中心计数：
            MAC = N * C_in * ∏S_in * (C_out/groups) * ∏K。

        计算逻辑：
            - 每个输出元素是长度 L 的点积：
                L = (C_in / groups) * ∏K
            - 输出元素总数：
                P = N * C_out * ∏S_out
            - 总乘加（FMA）：
                FMA = P * L
            - 若有偏置 B（p_shapes 第二个张量），每个输出元素再做 1 次加法：
                add += N * C_out * ∏S_out
        """

        # ---------- 取形状 ----------
        X = self.in0_shape  # 输入 X 形状 (N, C_in, ...)
        # 权重：Conv=(C_out, C_in/groups, *K)，ConvTranspose=(C_in, C_out/groups, *K)
        W = self.p_shapes[0]
        Y = self.out0_shape  # 输出 Y 形状 (N, C_out, *S_out) —— 一定存在

        N = X[0]
        C_in = X[1]
        C_out = Y[1]

        # 核尺寸列表 K（支持 1D/2D/3D：如 [kH, kW] 或 [kD, kH, kW]）
        K_dims = list(map(int, W[2:]))  # e.g. [kH, kW]
        # 输入/输出空间尺寸（如 [H_in, W_in] / [H_out, W_out]）
        S_in_dims = list(map(int, X[2:]))
        S_out_dims = list(map(int, Y[2:]))

        # group - INT (default is '1')
        groups = int(self._get_arg("group", 1))

        # ---------- 核体积与输出元素总数 ----------
        kernel_vol = 1
        for k in K_dims:
            kernel_vol *= k

        # 输出元素总数（Conv/ConvTranspose 一致，用于偏置加法计数）
        out_spatial_elems = 1
        for s in S_out_dims:
            out_spatial_elems *= s

        # ---------- 合法性：按 op_type 分支（两者权重布局不同） ----------
        if op_type == "ConvTranspose":
            # ConvTranspose 权重为 (C_in, C_out/groups, *K)：W[0]==C_in 且 W[1]*groups==C_out
            if len(W) >= 2 and (W[0] != C_in or W[1] * groups != C_out):
                issue(
                    "op_shape",
                    f"ConvTranspose: 权重({W[0]}, {W[1]}) 与 groups({groups})、"
                    f"C_in({C_in})/C_out({C_out}) 不一致",
                    op_type=op_type,
                )
                return
            if C_out % groups != 0:
                issue(
                    "op_shape",
                    f"ConvTranspose: C_out({C_out}) 无法被 groups({groups}) 整除",
                    op_type=op_type,
                )
                return
        else:
            # Conv 权重为 (C_out, C_in/groups, *K)：W[1]*groups==C_in
            if len(W) >= 2 and W[1] * groups != C_in:
                issue(
                    "op_shape",
                    f"Conv: 权重 C_in_per_group({W[1]}) * groups({groups}) != C_in({C_in})",
                    op_type=op_type,
                )
                return
            if C_in % groups != 0:
                issue(
                    "op_shape",
                    f"Conv: C_in({C_in}) 无法被 groups({groups}) 整除",
                    op_type=op_type,
                )
                return

        # ---------- 位置数 P 与点积长度 L ----------
        if op_type == "ConvTranspose":
            # 转置卷积以“输入元素”为中心计数：每个输入元素 x[n, ci, si] 与
            # (C_out/groups)*∏K 个权重相乘并累加到输出，故
            #   MAC = N * C_in * ∏S_in * (C_out/groups) * ∏K
            # （等价于把转置卷积看成对偶的普通卷积，输入/输出角色互换。）
            in_spatial_elems = 1
            for s in S_in_dims:
                in_spatial_elems *= s
            L = (C_out // groups) * kernel_vol
            P = N * C_in * in_spatial_elems
        else:
            # 普通卷积：每个输出元素是长度 L 的点积
            #   L = (C_in/groups) * ∏K，P = N * C_out * ∏S_out
            L = (C_in // groups) * kernel_vol
            P = N * C_out * out_spatial_elems

        # ---------- 乘加次数（FMA） ----------
        self.ci.fma += P * L

        # ---------- 偏置加法（可选） ----------
        # 若提供了 B（p_shapes[1]），每个输出元素再做 1 次加法
        if len(self.p_shapes) > 1:
            self.ci.add += N * C_out * out_spatial_elems

    def rnn(self, op_type, in_size=None, out_size=None):
        def _act_cost(act, per_hidden_one_dir, n=1):
            for _ in range(n):
                (
                    self.activation(act, None, per_hidden_one_dir)
                    if act != "Tanh"
                    else self.elementwise("Tanh", None, per_hidden_one_dir)
                )

        X_shape = self.in0_shape  # 3-D 输入
        # layout - INT (default is '0')
        layout = self._get_arg("layout", 0)
        if layout == 0:
            # X: [T, B, Din]
            seq_len, batch, in_size = X_shape[0], X_shape[1], X_shape[2]
        else:
            # X: [B, T, Din]
            batch, seq_len, in_size = X_shape[0], X_shape[1], X_shape[2]

        # hidden_size - INT
        hidden_size = self._get_arg("hidden_size", required=True)
        # direction - STRING (default is 'forward')
        direction = self._get_arg("direction", "forward")
        num_directions = 2 if direction == "bidirectional" else 1

        # activations：RNN 1/2 个，GRU 2/4 个，LSTM 3/6 个
        # RNN: One (or two if bidirectional) activation function for input gate.
        # LSTM: A list of 3 (or 6 if bidirectional) activation functions for input, output, forget, cell, and hidden.
        # GRU: A list of 2 (or 4 if bidirectional) activation functions for update, reset, and hidden gates.
        # The activation functions must be one of the activation functions specified above.
        # Optional: See the equations for default if not specified.
        if op_type == "RNN":
            default_activations = ["Tanh"]
        elif op_type == "LSTM":
            default_activations = ["Sigmoid", "Tanh", "Tanh"]
        elif op_type == "GRU":
            default_activations = ["Sigmoid", "Tanh"]
        else:
            issue("unknown_subop", f"Unknown RNN type: {op_type}", op_type=op_type)
            return
        default_activations *= num_directions
        activations = self._get_arg("activations", default_activations)

        # 局部修正：schema 默认值压制问题（仅 RNN 受影响）
        # ------------------------------------------------------------
        # 背景：get_node_attrs_dict() 会先用 ONNX schema 的默认值填充 op_args
        # （见 graph/utils.py::get_default_attrs）。而 RNN 在 schema 里为
        # activations 声明了具体默认值 ["Tanh", "Tanh"]（隐含按双向假设），
        # 该值被注入 op_args 后，get_arg() 会误认为“用户已显式指定”，从而
        # 压制此处按 direction 计算出的 default_activations —— 导致单向 RNN
        # 的激活数量从 1 被放大为 2，激活成本翻倍。
        # 说明：LSTM/GRU 的 activations 在 schema 中无真实默认值（UNDEFINED），
        # 本该落到 default_activations，不受此问题影响，故此处仅处理 RNN。
        # 处理：RNN 每个方向恒为 1 个激活，合法数量应等于 num_directions；
        # 若取到的数量与之不符，即判定为 schema 注入值，回退到按方向推算的默认值。
        if op_type == "RNN" and len(activations) != num_directions:
            activations = default_activations

        # clip：若非None，表示对每个门/线性和的“激活输入”做裁剪
        # Cell clip threshold. Clipping bounds the elements of a tensor in the range of [-threshold, +threshold]
        # and is applied to the input of activations. No clip if not specified.
        clip_thres = self._get_arg("clip", None)

        # 便捷计数：每方向/全部方向的逐元素数
        per_hidden_all_dirs = seq_len * batch * num_directions * hidden_size
        per_hidden_one_dir = seq_len * batch * hidden_size

        # ========== 3) 各算子分支 ==========
        if op_type == "RNN":
            """
            RNN（单门，默认激活 f=Tanh）
            ----------------------------
            公式（逐时间步）：
                H_t = f( X_t * W^T + H_{t-1} * R^T + Wb + Rb )

            权重形状（每方向）：
                W: [H, Din]        # 输入到隐层
                R: [H, H]          # 隐层到隐层
                B: [2*H] (可选)    # 连接前的偏置 [Wb, Rb] 逐元素相加

            flops 推导（所有时间步 & 样本 & 方向的总和）：
            1) 线性层（Wx + Rh）：
            - 对每个方向，每个时间步、每个样本，需要计算：
                X_t @ W^T  => Din * H FMA
                H_{t-1} @ R^T => H * H FMA
                合计 (Din + H) * H FMA
            - 总计：T * B * num_directions * (Din + H) * H
            2) 偏置加法（若 B 存在）：
            - 每个元素加一次 => T * B * num_directions * H
            3) 激活前 clip（若设置）：
            - 对每个 pre-activation 元素裁剪 => 同上元素数
            4) 激活 f：
            - 若 activations 提供了逐方向名称，则累加每方向的成本；
                否则默认 Tanh
            """
            has_B = len(self.p_shapes) >= 3  # X,W,R,B,...

            # (1) 线性层 Wx + Rh
            self.ci.fma += (
                seq_len * batch * num_directions * (in_size + hidden_size) * hidden_size
            )

            # (2) 偏置（B 是否提供）
            if has_B:
                self.ci.add += per_hidden_all_dirs

            # (3) 激活前 clip
            if clip_thres is not None:
                self.ci.logic += per_hidden_all_dirs * 2

            # (4) 激活（按方向或默认）
            for act in activations:
                _act_cost(act, per_hidden_one_dir, 1)

        elif op_type == "LSTM":
            """
            LSTM（四门：i, f, c(g), o；默认 f=Sigmoid, g=Tanh, h=Tanh）
            ----------------------------------------------------------
            公式（逐时间步，逐方向）：
                i_t = f( X_t*W_i^T + H_{t-1}*R_i^T + P_i ⊙ C_{t-1} + Wb_i + Rb_i )
                f_t = f( X_t*W_f^T + H_{t-1}*R_f^T + P_f ⊙ C_{t-1} + Wb_f + Rb_f )
                c~_t= g( X_t*W_c^T + H_{t-1}*R_c^T + Wb_c + Rb_c )
                C_t = f_t ⊙ C_{t-1} + i_t ⊙ c~_t
                o_t = f( X_t*W_o^T + H_{t-1}*R_o^T + P_o ⊙ C_t + Wb_o + Rb_o )
                H_t = o_t ⊙ h(C_t)

            权重形状（每方向）：
                W: [4H, Din]
                R: [4H, H]
                B: [8H] (可选)  # [Wb(iofc), Rb(iofc)]
                P: [3H] (可选)  # peephole：i,f 用 C_{t-1}；o 用 C_t

            属性：
                input_forget = 1 => i = 1 - f（省去 i 门的线性与激活；补 1 次逐元素减法）

            flops 概览：
                线性层：T * B * D * gates_linear * (Din + H) * H
                偏置：  T * B * D * gates_linear * H
                clip：  T * B * D * gates_linear * H * 2
                P门：   per_hidden_all_dirs * 3 mul + 3 add
                门激活：按 activations 或默认 (f,g) 计费（逐方向求和）
                C/H更新：3 mul + 1 add；外加 h(C_t) 的激活按 activations[h]
                input_forget=1: i = 1 - f
            """
            # input_forget - INT (default is '0')
            input_forget = self._get_arg("input_forget", 0)
            has_B = len(self.p_shapes) >= 4  # X,W,R,B,...
            has_P = len(self.p_shapes) >= 5  # ...,P

            # (1) 线性层：四门（或三门）* (Din + H) * H
            gates_linear = 3 if input_forget == 1 else 4
            self.ci.fma += (
                seq_len
                * batch
                * num_directions
                * gates_linear
                * (in_size + hidden_size)
                * hidden_size
            )

            # (2) 偏置
            if has_B:
                self.ci.add += (
                    seq_len * batch * num_directions * gates_linear * hidden_size
                )

            # (3) 激活前 clip（对门 pre-activation）
            if clip_thres is not None:
                self.ci.logic += (
                    seq_len * batch * num_directions * gates_linear * hidden_size * 2
                )

            # (4) peephole：i,f 用 C_{t-1}；o 用 C_t
            if has_P:
                self.ci.mul += per_hidden_all_dirs * 3  # i,f,o 各 1 次
                self.ci.add += per_hidden_all_dirs * 3

            # (5) 门激活：i/f/o 用 f；c(g) 用 g —— 按 activations 或默认
            # 单向 => [f, g, h]；双向 => [f_fwd, g_fwd, h_fwd, f_bwd, g_bwd, h_bwd]
            for i, act in enumerate(activations):
                if i % 3 == 0:  # f：用于 i/f/o
                    _act_cost(act, per_hidden_one_dir, 3)
                elif i % 3 == 1:  # g：用于 c~
                    _act_cost(act, per_hidden_one_dir, 1)

            # (6) 细胞/隐态更新（乘加不变）+ 输出激活 h(C_t) 计费（按 activations[h] 或默认）
            self.ci.mul += per_hidden_all_dirs * 3  # f⊙C_{t-1}、i⊙c~、o⊙h(C_t)
            self.ci.add += per_hidden_all_dirs * 1  # C_t 的一次加法
            for i, act in enumerate(activations):
                if i % 3 == 2:  # h：用于 h(C_t)
                    _act_cost(act, per_hidden_one_dir, 1)

            # (7) input_forget=1：i = 1 - f
            if input_forget == 1:
                self.ci.sub += per_hidden_all_dirs

        elif op_type == "GRU":  # op_type == "GRU"
            """
            GRU（三门：z, r, h~；默认 f=Sigmoid, g=Tanh）
            ------------------------------------------
            公式（逐时间步，逐方向）：
                z_t = f( X_t*W_z^T + H_{t-1}*R_z^T + Wb_z + Rb_z )
                r_t = f( X_t*W_r^T + H_{t-1}*R_r^T + Wb_r + Rb_r )
                # linear_before_reset = 0（默认）：
                h~_t = g( X_t*W_h^T + (r_t ⊙ H_{t-1})*R_h^T + Wb_h + Rb_h )
                # linear_before_reset = 1：
                h~_t = g( X_t*W_h^T + r_t ⊙ (H_{t-1}*R_h^T + Rb_h) + Wb_h )
                H_t  = (1 - z_t) ⊙ h~_t + z_t ⊙ H_{t-1}

            权重形状（每方向）：
                W: [3H, Din]
                R: [3H, H]
                B: [6H] (可选)  # [Wb(zrh), Rb(zrh)]

            flops 概览：
                线性层：T * B * D * 3 * (Din + H) * H
                偏置：  T * B * D * 3 * H
                clip：  per_hidden_all_dirs * 3 * 2
                门激活：z/r 用 f，h~ 用 g（按 activations 或默认）
                reset乘法：per_hidden_all_dirs * 1
                输出融合：2 mul + 1 add
            """
            # linear_before_reset - INT (default is '0')
            linear_before_reset = int(self._get_arg("linear_before_reset", 0))
            has_B = len(self.p_shapes) >= 3  # X,W,R,B,...

            # (1) 三门线性 Wx + Rh
            self.ci.fma += (
                seq_len
                * batch
                * num_directions
                * 3
                * (in_size + hidden_size)
                * hidden_size
            )

            # (2) 偏置
            if has_B:
                self.ci.add += per_hidden_all_dirs * 3
                # LBR=1 时，h~ 分支里还会在 Rh^T 之后、与 r 相乘之前加一次 Rb_h，
                # 我们在粗粒度模型中补记这一次加法（默认“每门1次”会漏掉它）。
                if linear_before_reset == 1:
                    self.ci.add += per_hidden_all_dirs  # 额外 +Rb_h

            # (3) 激活前 clip（z/r/h 三个 pre-activation）
            if clip_thres is not None:
                self.ci.logic += per_hidden_all_dirs * 3 * 2

            # (4) 门激活：按 activations 或默认
            # 单向 => [f, g]；双向 => [f_fwd, g_fwd, f_bwd, g_bwd]
            for i, act in enumerate(activations):
                if i % 2 == 0:  # z, r 用 f（2 次）
                    _act_cost(act, per_hidden_one_dir, 2)
                else:  # h~ 用 g（1 次）
                    _act_cost(act, per_hidden_one_dir, 1)

            # (5) reset 相关逐元素乘法（两个分支本质相同：1 次逐元素乘）
            # LBR=0：先 r⊙H_{t-1} 再 *R_h^T
            # LBR=1：先 H_{t-1}*R_h^T(+Rb_h) 再 ⊙r
            self.ci.mul += per_hidden_all_dirs * 1

            # (6) 最终状态融合
            self.ci.mul += per_hidden_all_dirs * 2
            self.ci.add += per_hidden_all_dirs * 1

        else:
            issue("unknown_subop", f"Unknown RNN operation: {op_type}", op_type=op_type)

    def attention(self, op_type, in_size=None, out_size=None):
        """
        Scaled dot-product attention cost model.

        Assumptions / inputs order:
        - self.in_shapes[0] = Q
        - self.in_shapes[1] = K
        - self.in_shapes[2] = V
        - Additional optional inputs (mask, past_k, past_v, ...) may exist but
            we assume K/V shapes already reflect the total sequence length (including past)
            where appropriate. We still respect qk_matmul_output_mode to avoid double counting.
        """
        # ---- shapes and basic head derivation ----
        q_shape, k_shape, v_shape = (
            self.in0_shape,
            self.in_shapes[1] if len(self.in_shapes) > 1 else None,
            self.in_shapes[2] if len(self.in_shapes) > 2 else None,
        )
        if not (q_shape and k_shape and v_shape):
            return

        # Determine whether inputs are 4D [B,H,S,head] or 3D [B,S,D]
        def parse_qkv(shape, role):
            if len(shape) == 4:
                B, H, S, Dhead = shape
                return {
                    "B": B,
                    "H": H,
                    "S": S,
                    "Dhead": Dhead,
                    "D": H * Dhead,
                    "is_4d": True,
                }
            elif len(shape) == 3:
                B, S, D = shape
                return {
                    "B": B,
                    "H": None,
                    "S": S,
                    "Dhead": None,
                    "D": D,
                    "is_4d": False,
                }
            else:
                issue(
                    "op_shape",
                    f"{op_type}: unsupported {role} shape {shape}",
                    op_type=op_type,
                )
                return None

        q = parse_qkv(q_shape, "Q")
        k = parse_qkv(k_shape, "K")
        v = parse_qkv(v_shape, "V")
        if not (q and k and v):
            return

        # Basic checks and derived dims
        B = q["B"]
        Sq = q["S"]
        # Derive q_num_heads and kv_num_heads
        q_num_heads = self._get_arg(
            "q_num_heads",
            q["H"] if q["H"] is not None else 1,
            required=True,
        )
        kv_num_heads = self._get_arg(
            "kv_num_heads",
            k["H"] if k["H"] is not None else q_num_heads,
            required=True,
        )

        # Per-head dims
        if q["is_4d"]:
            Dh = q["Dhead"]
        else:
            if not (q_num_heads > 0 and (q["D"] % q_num_heads == 0)):
                issue(
                    "op_shape",
                    f"attention: Q 嵌入维 {q['D']} 无法被 q_num_heads={q_num_heads} 整除",
                    op_type=op_type,
                )
                return
            Dh = q["D"] // q_num_heads

        if v["is_4d"]:
            Dv = v["Dhead"]
        else:
            if not (kv_num_heads > 0 and (v["D"] % kv_num_heads == 0)):
                issue(
                    "op_shape",
                    f"attention: V 嵌入维 {v['D']} 无法被 kv_num_heads={kv_num_heads} 整除",
                    op_type=op_type,
                )
                return
            Dv = v["D"] // kv_num_heads

        # K sequence length (Sk). Prefer K shape directly; extend with cache if provided.
        Sk = k["S"]
        if len(self.in_shapes) > 4 and self.in_shapes[4] is not None:  # past_key
            pk_shape = self.in_shapes[4]
            past_Sk = pk_shape[2] if len(pk_shape) == 4 else pk_shape[1]
            Sk += past_Sk

        # Number of heads used in QK matmul
        H = q_num_heads
        # Divisibility for 3D K as well
        if (not k["is_4d"]) and (k["D"] % kv_num_heads != 0):
            issue(
                "op_shape",
                f"attention: K 嵌入维 {k['D']} 无法被 kv_num_heads={kv_num_heads} 整除",
                op_type=op_type,
            )
            return

        # ---- options / flags ----
        has_mask = len(self.in_shapes) > 3 and self.in_shapes[3] is not None
        # is_causal - INT (default is '0')
        is_causal = bool(self._get_arg("is_causal", 0))
        # qk_matmul_output_mode - INT (default is '0')
        qk_mode = int(self._get_arg("qk_matmul_output_mode", 0))
        default_scale = 1.0 / math.sqrt(max(1, Dh))
        # scale - FLOAT Default value is 1/sqrt(head_size)
        scale = float(self._get_arg("scale", default_scale))
        # softcap - FLOAT (default is '0.0')
        softcap = float(self._get_arg("softcap", 0.0))

        # ---- Pre-scale Q and K by sqrt(scale) once (if scale != 1.0) ----
        if scale != 1.0:
            q_elems = B * Sq * (H * Dh)
            k_elems = B * Sk * (kv_num_heads * Dh)
            self.ci.mul += q_elems + k_elems

        # ---- Step 1: Q @ K^T  (per-head matmul) ----
        self.ci.fma += B * H * Sq * Sk * Dh

        # ---- Step 1.5: Mask (if not fused at or before mode 1) ----
        if (qk_mode < 1) and (has_mask or is_causal):
            self.ci.add += B * H * Sq * Sk

        # ---- Step 1.6: Softcap (if enabled and not fused up to mode 2) ----
        if (softcap > 0.0) and (qk_mode < 2):
            rows = B * H * Sq
            cols = Sk
            self.ci.div += rows * cols
            self.ci.hyperbolic += rows * cols
            self.ci.mul += rows * cols

        # ---- Step 2: Softmax (stable implementation) ----
        if qk_mode < 3:
            rows = B * H * Sq
            cols = Sk
            if cols > 1:
                self.ci.logic += rows * (cols - 1)
            self.ci.sub += rows * cols
            self.ci.exp += rows * cols
            if cols > 1:
                self.ci.add += rows * (cols - 1)
            self.ci.div += rows * cols

        # ---- Step 3: probs @ V  (attention weighted sum) ----
        self.ci.fma += B * H * Sq * Sk * Dv

        # ---- No Step 4 (output projection) ----
        # 该算子规范未包含输出投影权重输入；若模型在算子外另有线性层，应在对应算子里单独计费。

        return

    def build(self):
        if self.op_type in ELEMENTWISE_OPS:
            self.elementwise(self.op_type)
        elif self.op_type in REDUCTION_OPS:
            self.reduction(self.op_type)
        elif self.op_type in TENSOR_OPS:
            self.tensor(self.op_type)
        elif self.op_type in NORMALIZATION_OPS:
            self.normalization(self.op_type)
        elif self.op_type in ACTIVATION_OPS:
            self.activation(self.op_type)
        elif self.op_type in MATRIX_OPS:
            self.matrix(self.op_type)
        elif self.op_type in CONV_OPS:
            self.conv(self.op_type)
        elif self.op_type in RNN_OPS:
            self.rnn(self.op_type)
        elif self.op_type in ATTENTION_OPS:
            self.attention(self.op_type)
        else:
            issue(
                "unknown_op",
                f"Unsupported operation type: {self.op_type}",
                op_type=self.op_type,
            )

        return self.ci
