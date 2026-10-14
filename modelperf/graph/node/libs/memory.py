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
from ..definition import MemoryAccessInfo
from ...utils import numel, safe_axis
from .args import get_arg
from ....diagnostics import issue


class MemoryAccess:
    def __init__(
        self,
        op_type: str,
        op_args: Dict[str, Any],
        in_tensor_infos: List[Dict[str, Any]],
        out_tensor_infos: List[Dict[str, Any]],
        p_tensor_infos: List[Dict[str, Any]],
    ):
        # 基础信息
        self.op_type = op_type
        self.op_args = op_args

        # 缺省shape：空列表
        self.default_shape = []

        # 缺省dtype：根据输入
        if in_tensor_infos:
            self.default_dtype_size = in_tensor_infos[0]["dtype_size"]
        elif out_tensor_infos:
            self.default_dtype_size = out_tensor_infos[0]["dtype_size"]
        else:
            self.default_dtype_size = 4

        # 输入输出信息
        self.in_infos = in_tensor_infos
        self.p_infos = p_tensor_infos
        self.out_infos = out_tensor_infos

        # 输入输出形状
        self.in_shapes = [self._shape(ti) for ti in self.in_infos]
        self.p_shapes = [self._shape(ti) for ti in self.p_infos]
        self.out_shapes = [self._shape(ti) for ti in self.out_infos]

        # 输入输出数据类型大小
        self.in_dtype_sizes = [self._dtype_size(ti) for ti in self.in_infos]
        self.p_dtype_sizes = [self._dtype_size(ti) for ti in self.p_infos]
        self.out_dtype_sizes = [self._dtype_size(ti) for ti in self.out_infos]

        # 内存访问信息定义
        self.mi = MemoryAccessInfo()

        # 原始数据
        self.mi.in_tensor_infos = in_tensor_infos
        self.mi.p_tensor_infos = p_tensor_infos
        self.mi.out_tensor_infos = out_tensor_infos

        # 基础读写
        self.mi.read_act = sum(
            self._bytes(shape, dtype)
            for shape, dtype in zip(self.in_shapes, self.in_dtype_sizes)
        )
        self.mi.read_weight = sum(
            self._bytes(shape, dtype)
            for shape, dtype in zip(self.p_shapes, self.p_dtype_sizes)
        )
        self.mi.write_act = sum(
            self._bytes(shape, dtype)
            for shape, dtype in zip(self.out_shapes, self.out_dtype_sizes)
        )

        # 临时读写
        self.mi.read_temp = 0
        self.mi.write_temp = 0

    # ------------ 工具 ------------
    def _shape(self, info):
        return info["shape"] if info else self.default_shape

    def _dtype_size(self, info):
        return info["dtype_size"] if info else self.default_dtype_size

    def _bytes(self, shape, dtype_size):
        return numel(shape) * dtype_size

    def _set_base_io(self, read_act=None, read_weight=None, write_act=None):
        if read_act is not None:
            self.mi.read_act = read_act
        if read_weight is not None:
            self.mi.read_weight = read_weight
        if write_act is not None:
            self.mi.write_act = write_act

    def _get_arg(self, key: str, default=None, required=False):
        return get_arg(
            self.op_args,
            key,
            default=default,
            required=required,
            op_type=self.op_type,
        )

    def _temp_once(self, shape, dtype_size):
        """一次写入 + 一次读取该中间量（最小一致模型）"""
        b = self._bytes(shape, dtype_size)
        self.mi.write_temp += b
        self.mi.read_temp += b

    # ------------ 各类算子 ------------
    def elementwise(self, op_type):
        """
        逐元素：默认 I/O 够用（读全部输入，写全部输出），**无额外中间量**。
        特例：
          - Mod：有多步，计 3 个中间张量（div, q=floor(div), mul=q*y）→ 各写一次读一次。
          - Clip：视作 in-place，默认基础 I/O 已够用，不加 temp。
          - Sign/Shrink：条件选择 in-place，不加 temp。
          - 逻辑/比较/位运算/三角/Exp/Log/Reciprocal 等：不加 temp。
          - Where/IsInf/IsNaN/NonZero：不加 temp（NonZero 的索引输出在 out_infos 中体现）。
        """
        if not self.out_shapes:
            return

        if op_type == "Mod":
            out0_shape = self.out_shapes[0]
            out0_dtype = self.out_dtype_sizes[0]

            # div, q=floor(div), mul=q*y : 三个中间张量
            self._temp_once(out0_shape, out0_dtype)  # div
            self._temp_once(out0_shape, out0_dtype)  # q
            self._temp_once(out0_shape, out0_dtype)  # mul

    def reduction(self, op_type):
        """
        按“中间变量：部分和/部分积/当前极值”等，计 1 次写 + 1 次读到 temp。
        """
        in0_shape, out0_shape = self.in_shapes[0], self.out_shapes[0]
        in0_dtype, out0_dtype = self.in_dtype_sizes[0], self.out_dtype_sizes[0]

        if op_type in [
            "ReduceMax",
            "ReduceMin",
            "ArgMax",
            "ArgMin",
            "GlobalMaxPool",
            "MaxPool",
        ]:
            # 中间保存“当前极值”（或当前最值索引），按输出形状计一次写一次读
            self._temp_once(out0_shape, out0_dtype)

        elif op_type in ["ReduceSum", "ReduceMean", "ReduceProd", "ReduceSumSquare"]:
            # 部分和 / 部分积
            self._temp_once(out0_shape, out0_dtype)

        elif op_type == "ReduceL1":
            # abs(x) 可流式；部分和
            self._temp_once(out0_shape, out0_dtype)

        elif op_type == "ReduceL2":
            # 中间：x^2 (in_size)，以及部分和(out_size)
            self._temp_once(in0_shape, in0_dtype)  # x^2
            self._temp_once(out0_shape, out0_dtype)  # sum

        elif op_type == "ReduceLogSum":
            # 部分和 + 对和取log（log输入直接用部分和，不另计）
            self._temp_once(out0_shape, out0_dtype)  # sum

        elif op_type == "ReduceLogSumExp":
            # exp(x) (in_size) + 部分和(out_size)
            self._temp_once(in0_shape, in0_dtype)  # exp(x)
            self._temp_once(out0_shape, out0_dtype)  # sum

        elif op_type == "TopK":
            # 中间K缓冲（与输出 K 对齐）
            self._temp_once(out0_shape, out0_dtype)

        elif op_type == "Unique":
            # 排序/哈希需要的辅助缓冲：按最坏“排序缓存”计一次输入大小
            self._temp_once(in0_shape, in0_dtype)

        elif op_type == "AveragePool":
            # 窗口内求和 -> 部分和 out_size
            self._temp_once(out0_shape, out0_dtype)

        elif op_type == "GlobalAveragePool":
            self._temp_once(out0_shape, out0_dtype)

        else:
            issue(
                "unknown_subop",
                f"Unknown reduction operation: {op_type}",
                op_type=op_type,
            )

    def tensor(self, op_type):
        """
        张量变换/构造类：对每一类做 I/O 修正，避免默认计数误差。
        """

        # ---- 生成类 ----
        if op_type in [
            "EyeLike",
            "RandomNormal",
            "RandomNormalLike",
            "RandomUniform",
            "RandomUniformLike",
            "Constant",
            "ConstantOfShape",
            "Range",
            "OneHot",
        ]:
            # - EyeLike
            # 2D input tensor to copy shape, and optionally, type information from.
            # Output tensor, same shape as input tensor.
            # - RandomNormal
            # Output tensor of random values drawn from normal distribution.
            # - RandomNormalLike
            # Input tensor to copy shape and optionally type information from.
            # Output tensor of random values drawn from normal distribution.
            # - RandomUniform
            # Output tensor of random values drawn from uniform distribution.
            # - RandomUniformLike
            # Input tensor to copy shape and optionally type information from.
            # Output tensor of random values drawn from uniform distribution.
            # - Constant
            # Output tensor containing the same value of the provided tensor.
            # - ConstantOfShape
            # 1D tensor. The shape of the expected output tensor.
            # If empty tensor is given, the output would be a scalar. All values must be >= 0.
            # Output tensor of shape specified by ‘input’.
            # - Range
            # A 1-D tensor with same type as the inputs containing generated range of values.
            # - OneHot
            # indices-input tensor containing indices; depth; values;
            # Tensor of rank one greater than input tensor ‘indices’.

            # like类算子不真正读输入，以及一些无输入算子
            in_read = (
                self.mi.read_act
                if op_type in ["ConstantOfShape", "Range", "OneHot"]
                else 0
            )
            self._set_base_io(
                read_act=in_read, read_weight=0, write_act=self.mi.write_act
            )
            return

        # ---- 纯视图 / 元数据 ----
        if op_type in ["Reshape", "Squeeze", "Unsqueeze", "Flatten", "Identity"]:
            # 视图：不搬数据；输出也可看作 view，这里保守计 0 写
            self._set_base_io(read_act=0, read_weight=0, write_act=0)
            return

        if op_type in ["Shape", "Size"]:
            # 由元数据构造输出：不读输入数据，只写输出
            # - Shape
            # An input tensor.
            # Shape of the input tensor
            # - Size
            # An input tensor.
            # Total number of elements of the input tensor

            self._set_base_io(read_act=0, read_weight=0, write_act=self.mi.write_act)
            return

        # ---- 需要重排/搬运 ----
        if op_type in ["Transpose", "DepthToSpace", "SpaceToDepth", "ReverseSequence"]:
            # 全量搬运：读完整输入 + 写完整输出；无 temp
            return

        # ---- 切片/拼接/扩展/复制/收集/散射 ----
        if op_type == "Slice":
            # 只读取切出的部分：读取量≈写出量（以输入 dtype）
            # input: data, starts, ends, axes, steps
            # output: Sliced data tensor.
            in0_shape, out0_shape = self.in_shapes[0], self.out_shapes[0]
            in0_dtype, out0_dtype = self.in_dtype_sizes[0], self.out_dtype_sizes[0]

            slice_read_bytes = self._bytes(out0_shape, in0_dtype)
            other_read_bytes = sum(
                self._bytes(shape, dtype)
                for shape, dtype in zip(self.in_shapes[1:], self.in_dtype_sizes[1:])
            )
            self._set_base_io(
                read_act=slice_read_bytes + other_read_bytes,
                read_weight=0,
                write_act=self._bytes(out0_shape, out0_dtype),
            )
            return

        if op_type == "Split":
            # 读入=整输入；写出=各输出总和；无 temp
            # input: The tensor to split, split: Optional length of each output.
            # output: One or more outputs forming list of tensors after splitting
            return

        if op_type == "Concat":
            # 读入=各输入总和；写出=整输出；无 temp
            # List of tensors for concatenation
            # Concatenated tensor
            return

        if op_type == "Expand":
            # 广播展开：读一次输入，写完整输出；无 temp
            # input: Input tensor
            # shape: A 1-D tensor indicates the shape you want to expand to, following the broadcast rule
            # Output tensor
            # NOTE， 虽然输入被读取了多次，但是是广播读取，所以记一次
            return

        if op_type == "Tile":
            # input: Input tensor of any shape.
            # repeats: 1D int64 tensor of the same length as input’s dimension number,
            # includes numbers of repeated copies along input’s dimensions.
            # output: Output tensor of the same dimensions and type as tensor input. output_dim[i] = input_dim[i] * repeats[i]
            # NOTE, 真正重复读取了，重复读取次数 = |Y| / |X|（向下取整，至少1）
            in0_shape, out0_shape = self.in_shapes[0], self.out_shapes[0]
            in0_dtype, out0_dtype = self.in_dtype_sizes[0], self.out_dtype_sizes[0]

            in_bytes = self._bytes(in0_shape, in0_dtype)
            out_bytes = self._bytes(out0_shape, out0_dtype)
            repeats = max(1, numel(out0_shape) // max(1, numel(in0_shape)))

            self._set_base_io(
                read_act=in_bytes * repeats,
                read_weight=0,
                write_act=out_bytes,
            )
            return

        if op_type == "Pad":
            # 读入=整输入；写出=整输出
            # input: data, pads, constant_value, axes
            # output: Tensor after padding.
            # NOTE, mode=constant, 填充值来自常量，输入读一次
            # mode in ('reflect', 'edge'), 填充值来自边界, 理想情况广播读（每元素只读一次）
            return

        if op_type in ["Gather", "GatherElements", "GatherND"]:
            # data: Tensor of rank r >= 1.
            # indices: Tensor of int32/int64 indices, of any rank q.
            # output: Tensor of rank q + (r - 1).
            # NOTE, TODO, 读取的数据为indices + 输出对应的输入元素， 写入的数据为输出元素

            indices_read = self._bytes(self.in_shapes[1], self.in_dtype_sizes[1])
            read_act = self.mi.write_act
            self._set_base_io(
                read_act=indices_read + read_act,
                read_weight=0,
                write_act=self.mi.write_act,
            )
            return

        if op_type in ["Scatter", "ScatterElements", "ScatterND"]:
            # data: Tensor of rank r >= 1.
            # indices: Tensor of int32/int64 indices, of r >= 1 (same rank as input).
            # updates: Tensor of rank r >=1 (same rank and shape as indices)
            # output: Tensor of rank r >= 1 (same rank as input).
            # NOTE, TODO, 读取的数据为indices + updates, 写入的数据为输出元素 (updates 对应的元素)

            indices_read = self._bytes(self.in_shapes[1], self.in_dtype_sizes[1])
            updates_read = self._bytes(self.in_shapes[2], self.in_dtype_sizes[2])
            write_act = self._bytes(self.in_shapes[2], self.in_dtype_sizes[2])
            self._set_base_io(
                read_act=indices_read + updates_read,
                read_weight=0,
                write_act=write_act,
            )
            return

        if op_type in ["Resize", "Upsample"]:
            # - Resize
            # X, roi, scale, sizes
            # Y: N-D tensor after resizing
            # - Upsample
            # X, scales
            # Y: N-D tensor after resizing
            X, Y = self.in_shapes[0], self.out_shapes[0]
            in_dtype, out_dtype = self.in_dtype_sizes[0], self.out_dtype_sizes[0]

            N = len(X)
            # mode - STRING (default is 'nearest')
            mode = self._get_arg("mode", "nearest")
            # axes - INTS
            # If provided, it specifies a subset of axes that ‘roi’, ‘scales’ and ‘sizes’ refer to.
            # If not provided, all axes are assumed [0, 1, …, r-1], where r = rank(data).
            axes = self._get_arg("axes", list(range(N)))
            axes = [safe_axis(axis, N) for axis in axes]
            s = [(Y[i] / X[i]) if i in axes else 1.0 for i in range(N)]

            if op_type == "Resize":
                # antialias - INT (default is '0')
                antialias = self._get_arg("antialias", 0)
                base = {"nearest": 1, "linear": 2, "cubic": 4}.get(mode, 1)
                taps_per_axis = []
                for i in range(N):
                    if i not in axes:
                        taps_per_axis.append(1)
                        continue
                    si = max(float(s[i]), 1e-12)
                    if antialias and mode in ("linear", "cubic") and si < 1.0:
                        taps_per_axis.append(math.ceil(base * (1.0 / si)))
                    else:
                        taps_per_axis.append(base)
                T = numel(taps_per_axis)

            else:  # Upsample
                # 与 Resize 同口径: 仅对真正被缩放的轴计 tap, 未缩放轴(含 batch/channel)计 1
                base = {"nearest": 1, "linear": 2, "cubic": 4}.get(mode, 1)
                taps_per_axis = [
                    base if (i in axes and abs(float(s[i]) - 1.0) > 1e-12) else 1
                    for i in range(N)
                ]
                T = numel(taps_per_axis)

            input_sampling_reads = self._bytes(Y, in_dtype) * T
            self._set_base_io(
                read_act=input_sampling_reads,
                read_weight=self.mi.read_weight,
                write_act=self._bytes(Y, out_dtype),
            )
            return

        # 其他张量类操作：默认
        return

    def normalization(self, op_type):
        """
        归一化：依据“中间变量”口径，给每个中间向量计 1 次写 + 1 次读。
        """
        out0_dtype = self.out_dtype_sizes[0]

        if op_type == "BatchNormalization":
            # training_mode - INT(default is "0")
            training_mode = self._get_arg("training_mode", 0)
            if training_mode and self.in_shapes and len(self.in_shapes[0]) >= 2:
                C = self.in_shapes[0][1]
                if C > 0:
                    reduced_shape = [C]
                    self._temp_once(reduced_shape, out0_dtype)  # cur_mean
                    self._temp_once(reduced_shape, out0_dtype)  # cur_var
            return

        if op_type == "LayerNormalization":
            if self.in_shapes and len(self.in_shapes[0]) >= 2:
                # axis - INT (default is '-1')
                axis = safe_axis(self._get_arg("axis", -1), len(self.in_shapes[0]))
                reduced_elems = numel(self.in_shapes[0][:axis])
                if reduced_elems > 0:
                    reduced_shape = [reduced_elems]
                    self._temp_once(reduced_shape, out0_dtype)  # mean
                    self._temp_once(reduced_shape, out0_dtype)  # var
            return

        if op_type == "InstanceNormalization":
            if self.in_shapes and len(self.in_shapes[0]) >= 2:
                N, C = self.in_shapes[0][0], self.in_shapes[0][1]
                reduced_shape = [N * C] if (N * C) > 0 else []
                if reduced_shape:
                    self._temp_once(reduced_shape, out0_dtype)  # mean
                    self._temp_once(reduced_shape, out0_dtype)  # var
            return

        if op_type == "GroupNormalization":
            if self.in_shapes:
                N = self.in_shapes[0][0]
                # num_groups - INT(required)
                G = self._get_arg("num_groups", required=True)
                reduced_shape = [N * G] if (N * G) > 0 else []
                if reduced_shape:
                    self._temp_once(reduced_shape, out0_dtype)  # mean
                    self._temp_once(reduced_shape, out0_dtype)  # var
            return

        if op_type == "RMSNormalization":
            if self.in_shapes:
                # axis - INT (default is '-1')
                axis = safe_axis(self._get_arg("axis", -1), len(self.in_shapes[0]))
                reduced_elems = numel(self.in_shapes[0][:axis])
                if reduced_elems > 0:
                    reduced_shape = [reduced_elems]
                    self._temp_once(reduced_shape, out0_dtype)  # mean(x^2)/RMS
            return

        if op_type == "LpNormalization":
            if self.in_shapes:
                # axis - INT (default is '-1')
                axis = safe_axis(self._get_arg("axis", -1), len(self.in_shapes[0]))
                reduced_elems = numel(self.in_shapes[0][:axis])
                if reduced_elems > 0:
                    reduced_shape = [reduced_elems]
                    self._temp_once(reduced_shape, out0_dtype)  # norm
            return

        if op_type == "MeanVarianceNormalization":
            # axes - INTS (default is ['0', '2', '3'])
            axes = self._get_arg("axes", [0, 2, 3])
            axes = [safe_axis(a, len(self.in_shapes[0])) for a in axes]
            if self.in_shapes:
                dims = self.in_shapes[0]
                kept = [dims[ax] for ax in range(len(dims)) if ax not in axes]
                kept_elems = numel(kept)
                if kept_elems > 0:
                    reduced_shape = [kept_elems]
                    self._temp_once(reduced_shape, out0_dtype)  # mean
                    self._temp_once(reduced_shape, out0_dtype)  # var
            return

        issue(
            "unknown_subop",
            f"Unknown normalization operation: {op_type}",
            op_type=op_type,
        )

    def activation(self, op_type):
        """
        激活：大多无中间；Softmax/LogSoftmax 按稳定实现计：
          - max(x)  (out_size 形状)
          - exp(x-max) (与输入等大)
          - sum(exp(...)) (out_size 形状)
        """
        in0_shape, out0_shape = self.in_shapes[0], self.out_shapes[0]
        in0_dtype, out0_dtype = self.in_dtype_sizes[0], self.out_dtype_sizes[0]

        if op_type in ["Softmax", "LogSoftmax"]:
            if out0_shape:
                self._temp_once(out0_shape, out0_dtype)  # max(x)
            if in0_shape:
                self._temp_once(in0_shape, in0_dtype)  # exp(x - max)
            if out0_shape:
                self._temp_once(out0_shape, out0_dtype)  # sum(exp(...))

            return
        # 其他激活：无额外中间量
        return

    def matrix(self, op_type):
        """
        MatMul / Gemm：默认 I/O 即可（读取 A/B/(C)，写 Y），无额外中间。
        """
        return

    def conv(self, op_type):
        """
        Conv / ConvTranspose：读取 X/W/(B)，写 Y；无额外中间。

        workspace 计一次隐式 im2col 缓冲，形状 [N*(C_in/group)*prod(kernel), prod(spatial_out)]，
        支持任意空间维数（1D/2D/3D）。
        """
        in_shape, out_shape = self.in_shapes[0], self.out_shapes[0]
        N, C_in = in_shape[0], in_shape[1]
        spatial_out = out_shape[2:]
        # kernel_shape - INTS (required)
        kernel = self._get_arg("kernel_shape", required=True)
        # group - INT (default is '1')
        groups = int(self._get_arg("group", 1))

        workspace = self._bytes(
            [N * (C_in // groups) * numel(list(kernel)), numel(spatial_out)],
            self.in_dtype_sizes[0],
        )
        self.mi.read_temp += workspace  # TODO
        self.mi.write_temp += workspace
        return

    def rnn(self, op_type):
        """
        RNN/LSTM/GRU：给门/中间态按“每隐藏元素一次写一次读”的最小一致计数。
        - RNN：1 个 pre-activation 缓冲
        - LSTM：i,f,c~,o 四个门（input_forget=1 => 三门）
        - GRU：z,r,h~ 三个门
        """
        if not self.in_shapes or len(self.in_shapes[0]) != 3:
            return
        if not self.out_shapes or not self.out_shapes[0]:
            return

        X_shape = self.in_shapes[0]
        # layout - INT (default is '0')
        layout = self._get_arg("layout", 0)
        if layout == 0:
            # X: [T, B, Din]
            seq_len, batch = X_shape[0], X_shape[1]
        else:
            batch, seq_len = X_shape[0], X_shape[1]

        # hidden_size - INT
        hidden_size = self._get_arg("hidden_size", required=True)
        # direction - STRING (default is 'forward')
        direction = self._get_arg("direction", "forward")
        num_directions = 2 if direction == "bidirectional" else 1

        per_hidden_all_dirs = seq_len * batch * num_directions * hidden_size
        if per_hidden_all_dirs <= 0:
            return
        buf_shape = [per_hidden_all_dirs]

        if op_type == "RNN":
            self._temp_once(buf_shape, self.in_dtype_sizes[0])  # 1 个 pre-activation

        elif op_type == "LSTM":
            # input_forget - INT (default is '0')
            input_forget = self._get_arg("input_forget", 0)
            gates = 3 if input_forget == 1 else 4
            for _ in range(gates):  # i,f,(c~),(o)
                self._temp_once(buf_shape, self.in_dtype_sizes[0])

        elif op_type == "GRU":
            for _ in range(3):  # z,r,h~
                self._temp_once(buf_shape, self.in_dtype_sizes[0])
        else:
            issue(
                "unknown_subop",
                f"Unknown RNN operation: {op_type}",
                op_type=op_type,
            )

    def attention(self, op_type):
        """
        Scaled Dot-Product Attention：
          中间量（最低一致）：
            - logits: [B,H,Sq,Sk]
            - rowwise max: [B,H,Sq]
            - rowwise sum: [B,H,Sq]
            - prob  (=softmax(logits)): [B,H,Sq,Sk] 需要参与后续 prob@V

        Q/K/V 支持 3D [B,S,D] 与 4D [B,H,S,Dhead] 两种布局, 与 computation
        侧 attention 保持一致: 4D 时 H 取自形状, 3D 时取自 q_num_heads 属性。
        """
        if len(self.in_shapes) < 3:
            return
        q_shape, k_shape, v_shape = (
            self.in_shapes[0],
            self.in_shapes[1],
            self.in_shapes[2],
        )
        if not q_shape or not k_shape or not v_shape:
            return
        if len(q_shape) not in (3, 4) or len(k_shape) not in (3, 4):
            issue(
                "op_shape",
                f"{op_type}: unsupported Q/K shape {q_shape} / {k_shape}",
                op_type=op_type,
            )
            return

        # 3D [B,S,D] 与 4D [B,H,S,Dhead] 均取倒数第二维为序列长度
        B, Sq = q_shape[0], q_shape[-2]
        if len(q_shape) == 4:
            H = self._get_arg("q_num_heads", q_shape[1], required=True)
        else:
            H = self._get_arg("q_num_heads", required=True)
        Sk = k_shape[-2]

        logits_shape = [B, H, Sq, Sk]
        rows_shape = [B, H, Sq]
        prob_shape = [B, H, Sq, Sk]

        self._temp_once(logits_shape, self.in_dtype_sizes[0])  # logits
        self._temp_once(rows_shape, self.in_dtype_sizes[0])  # rowwise max
        self._temp_once(rows_shape, self.in_dtype_sizes[0])  # rowwise sum
        self._temp_once(prob_shape, self.in_dtype_sizes[0])  # probabilities

    # ------------ 总入口 ------------
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

        return self.mi
