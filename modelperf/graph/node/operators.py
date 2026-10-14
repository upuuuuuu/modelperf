ELEMENTWISE_OPS = {
    # 算术运算
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Pow",
    "Sqrt",
    "Mod",
    "Neg",
    "Abs",
    "Reciprocal",
    "Ceil",
    "Floor",
    "Round",
    # 逻辑运算
    "Equal",
    "Greater",
    "GreaterOrEqual",
    "Less",
    "LessOrEqual",
    "And",
    "Or",
    "Xor",
    "Not",
    "BitwiseAnd",
    "BitwiseOr",
    "BitwiseXor",
    "BitwiseNot",
    # 超越函数 (三角函数/指数函数/对数函数)
    "Sin",
    "Cos",
    "Tan",
    "Asin",
    "Acos",
    "Atan",
    "Sinh",
    "Cosh",
    "Tanh",
    "Asinh",
    "Acosh",
    "Atanh",
    "Exp",
    "Log",
    "Erf",
    # 条件类
    "Where",
    "IsInf",
    "IsNaN",
    "NonZero",
    # 其它
    "Cast",
    "CastLike",
    "Sign",
    "Shrink",
    "Clip",
}


REDUCTION_OPS = {
    # 基础归约
    "ReduceMax",
    "ReduceMin",
    "ReduceSum",
    "ReduceMean",
    "ReduceProd",
    # 统计类
    "ReduceL1",
    "ReduceL2",
    "ReduceLogSum",
    "ReduceLogSumExp",
    "ReduceSumSquare",
    # 索引归约
    "ArgMax",
    "ArgMin",
    "TopK",
    "Unique",
    # 池化类
    "AveragePool",
    "MaxPool",
    "GlobalAveragePool",
    "GlobalMaxPool",
}


TENSOR_OPS = {
    # 视图操作
    "Reshape",
    "Squeeze",
    "Unsqueeze",
    "Flatten",
    "Expand",
    "Shape",
    "Size",
    "Identity",
    # 重排
    "Transpose",
    "DepthToSpace",
    "SpaceToDepth",
    "ReverseSequence",
    # 拼接与切分类
    "Slice",
    "Split",
    "Concat",
    # 复制与扩展类
    "Tile",
    "Pad",
    # 索引类
    "Gather",
    "GatherElements",
    "GatherND",
    "Scatter",
    "ScatterElements",
    "ScatterND",
    # 生成类
    "Constant",
    "ConstantOfShape",
    "EyeLike",
    "Range",
    "OneHot",
    "RandomNormal",
    "RandomNormalLike",
    "RandomUniform",
    "RandomUniformLike",
    # 采样与插值
    "Resize",
    "Upsample",
}


NORMALIZATION_OPS = {
    "BatchNormalization",
    "LayerNormalization",
    "GroupNormalization",
    "InstanceNormalization",
    "RMSNormalization",
    "LpNormalization",
    "MeanVarianceNormalization",
}


ACTIVATION_OPS = {
    "Relu",
    "ThresholdedRelu",
    "LeakyRelu",
    "PRelu",
    "Sigmoid",
    "Swish",
    "HardSigmoid",
    "HardSwish",
    "Softplus",
    "Softsign",
    "Elu",
    "Selu",
    "Gelu",
    "Softmax",
    "LogSoftmax",
}


CONV_OPS = {"Conv", "ConvTranspose"}
MATRIX_OPS = {"Gemm", "MatMul"}
RNN_OPS = {"GRU", "LSTM", "RNN"}
ATTENTION_OPS = {"Attention"}


# ============================================
# ONNX 算子按可学习性分类
# ============================================

# 1. 带可学习参数 (Learnable Ops)
LEARNABLE_OPS = {
    # 卷积
    "Conv",
    "ConvTranspose",
    "DeformConv",
    # 全连接 / 线性
    "Gemm",
    # 归一化层（有可学习 γ/β）
    "BatchNormalization",
    "LayerNormalization",
    "InstanceNormalization",
    "GroupNormalization",
    "RMSNormalization",
    # 参数化激活
    "PRelu",
    # 注意力机制
    "Attention",
    # 数学线性类（取决于输入是否为 Parameter）
    "MatMul",
    "MatMulInteger",
    "QLinearMatMul",
    "QLinearConv",
    "ConvInteger",
}

# 2. 可能带可学习参数 (Conditionally Learnable Ops)
CONDITIONALLY_LEARNABLE_OPS = {
    # 归一化类（若 γ/β 冻结则不可学习）
    "BatchNormalization",
    "LayerNormalization",
    "InstanceNormalization",
    "GroupNormalization",
    "RMSNormalization",
    # 参数化激活（α 可固定）
    "PRelu",
    # 注意力模块（若权重冻结）
    "Attention",
    # 可学习仿射变换
    "AffineGrid",
}

# 3. 永远不带可学习参数 (Non-learnable Ops)
NON_LEARNABLE_OPS = {
    # 数学 / 基本运算
    "Abs",
    "Acos",
    "Acosh",
    "Add",
    "Asin",
    "Asinh",
    "Atan",
    "Atanh",
    "Ceil",
    "Clip",
    "Cos",
    "Cosh",
    "Div",
    "Elu",
    "Equal",
    "Erf",
    "Exp",
    "Floor",
    "Greater",
    "GreaterOrEqual",
    "LeakyRelu",
    "Less",
    "LessOrEqual",
    "Log",
    "Mod",
    "Mul",
    "Neg",
    "Pow",
    "Reciprocal",
    "Relu",
    "Round",
    "Sigmoid",
    "Sign",
    "Sin",
    "Sinh",
    "Softmax",
    "Softplus",
    "Softsign",
    "Sqrt",
    "Sub",
    "Tan",
    "Tanh",
    "ThresholdedRelu",
    # 池化 / 归约
    "AveragePool",
    "MaxPool",
    "GlobalAveragePool",
    "GlobalMaxPool",
    "GlobalLpPool",
    "ReduceSum",
    "ReduceMean",
    "ReduceMax",
    "ReduceMin",
    "ReduceProd",
    "ReduceL1",
    "ReduceL2",
    "ReduceLogSum",
    "ReduceLogSumExp",
    "ReduceSumSquare",
    "ArgMax",
    "ArgMin",
    "Mean",
    # 归一化 (非参数类)
    "LpNormalization",
    "MeanVarianceNormalization",
    # 张量 / 形状操作
    "Concat",
    "Reshape",
    "Transpose",
    "Split",
    "Flatten",
    "Expand",
    "Slice",
    "Pad",
    "Squeeze",
    "Unsqueeze",
    "Tile",
    "DepthToSpace",
    "SpaceToDepth",
    # 控制 / 逻辑 / 随机
    "If",
    "Loop",
    "Scan",
    "Where",
    "And",
    "Or",
    "Not",
    "Xor",
    "Greater",
    "Less",
    "Equal",
    "Bernoulli",
    "RandomNormal",
    "RandomUniform",
    "RandomNormalLike",
    "RandomUniformLike",
    # 常量 / 初始化
    "Constant",
    "ConstantOfShape",
    "EyeLike",
    "Range",
    # 其它 / 信号 / 字符串
    "BitShift",
    "BitwiseAnd",
    "BitwiseNot",
    "BitwiseOr",
    "BitwiseXor",
    "HannWindow",
    "HammingWindow",
    "BlackmanWindow",
    "StringConcat",
    "StringSplit",
    "StringNormalizer",
    "DFT",
    "STFT",
    "MelWeightMatrix",
    "GridSample",
    "RoiAlign",
    "NonMaxSuppression",
    "TopK",
    "CumSum",
    "Trilu",
    "Det",
}
