"""量化图归一化: 把 ONNX Runtime 的 QDQ 表示折叠成"真位宽图"。

背景
----
ONNX 本身没有"算子精度"这一概念。ONNX Runtime 的 ``quantize_static`` 默认
输出 QDQ 格式: 每个被量化的算子被
``float -> QuantizeLinear(下称 Q) -> int8 -> DequantizeLinear(下称 DQ) -> float``
夹在中间, 权重被 ``int8 initializer -> DQ -> float`` 包裹。于是整张图在结构
上处处是 float, 下游 ModelPerf 会把**所有激活按 fp32 口径估访存**, 且权重
落在 DQ 输出 (非 initializer) 上而丢失, 量化图等价于 fp32 图: 既不准, 也
根本跑不起来 (Conv 的 ``W = p_shapes[0]`` 取不到而 IndexError)。

本模块在 ONNX 简化之后、IR 构建之前插入一个**前置 pass**, 把 Q/DQ 折叠成
真正按位宽区分的图: 量化算子的激活输入/输出直接引用 int8 张量名, 权重回到
int8 initializer, 冗余的 Q/DQ 节点删除; 仅在真实发生 float<->int8 过渡处
保留一个等价语义的 ``Cast`` 节点。全程不引入外部参数、dtype 只读图内声明、
对 fp32 模型零副作用 (天然幂等)。

判据 (单一区域规则)
------------------
不做任何来源分叉 (tf2onnx / ORT 一视同仁), 只按图内张量关系判定。

先取"真 int8 量化张量": 图内声明为 INT8/UINT8 的张量, 加 QuantizeLinear (Q)
的输出 (Q 的输出恒为 int8)。据此定义**归一化区域**:

  入口 = 真 DQ 节点的输出 (真 DQ: 其数据输入是真 int8 张量)
  出口 = 真 Q 节点的输入 (Q 一律视为真)
  区域 = 从入口前向可达 ∩ 能后向到达出口 的非 Q/DQ 节点集

区域内的算子即"被量化算子": 一段 ``DQ -> ... -> Q`` 的数据流, 无论中间夹
几个算子, 都落在同一区域内被一并折叠 -- 既覆盖 tf2onnx 的非对称布局
``DQ -> Conv -> Relu -> Q`` (中间算子输出不直接喂 Q), 也覆盖 ORT 的逐算子
紧包裹 ``DQ -> op -> Q``。区域外的算子按 fp32 处理 (被排除量化的 fp32 岛)。

伪 DQ (量化侧非真 int8, 如 tf2onnx 把 int32 zero_point / BN 常量包出的转换)
不构成入口, 也不参与区域。

折叠规则 (float 侧张量是否需要真实以 float 存在)
---------------------------------------------
对 Q 的 float 输入张量 F, 需要保留为 float 当且仅当
  (a) F 是图输入; 或
  (b) F 存在既非 Q/DQ 又不在区域内的消费者 (fp32 岛入口 / 真实 float 读取);
  (c) F 的生产者存在且是既非 Q/DQ 又不在区域内的算子 (fp32 生产算子); 或
  (d) F 无生产者或本身是常量 initializer。
对 DQ 的 float 输出张量 G, 需要保留为 float 当且仅当
  (a) G 是图输出; 或
  (b) G 存在既非 Q/DQ 又不在区域内的消费者 (fp32 岛入口)。
命中任一 => 保留该 Q/DQ 节点并**就地改写为 Cast** (张量名不变, 语义等价);
否则 => 折叠: 把 float 侧名字别名到 int8 侧名字 (float->int8 方向, 使 int8
侧成为规范名, 下游据此按 1 字节计访存), 并删除该节点。

条件 (c) 不可省: 实测 gnn 有 20 个 Q、transformer 有大量 Q 的 float 输入由
Gather / ScatterElements / Transpose / Unsqueeze / Relu / Div / ReduceMean
等 fp32 算子产出, 若一律折叠会把 fp32 算子的输出张量改名成 int8, 令其写出
访存被低估 4 倍。

别名方向
--------
统一按 **float 侧 -> int8 侧** 折叠 (Q: ``alias[input] = output``;
DQ: ``alias[output] = input``)。权重 initializer 永不被别名, 因此 ``resolve``
自然落到 int8 initializer 上; 激活链自然落到 ``yq``; 图 I/O 名永不被别名
(命中 (a) 即保留)。共享 float 张量的再量化链 (同一 DQ 输出同时喂 Q 与量化
算子) 采用"先到先得", 后到的节点就地保留为 Cast, 保证图完整。

Cast 而非 DQ 的理由
------------------
``Cast`` 已在 operators.py 注册, 访存行为与 Q/DQ 完全一致 (按 in/out 逐元素
拷贝, 读满源位宽、写满目标位宽), 且不会触发 ``unknown_op`` 告警; 而
``QuantizeLinear`` / ``DequantizeLinear`` 未注册, 直接保留会误报。

五族实测 (seer_v2 config, torch.onnx.export -> ORT quantize_static(QDQ,
per_channel=False, MinMax), 经 onnxsim 简化后本 pass 实际见到的图)
------------------------------------------------------------------
family       body  Q(=actDQ)  weightDQ  Q.out->DQ  DQ.out->图输出  Q.in_shared
cnn          124   125        96        100%       1             0
gnn          91    63         24        100%       1             4 (全为权重)
rnn          7     3          4         100%       1             0
transformer  953   651        100       100%       1             9 (全为权重)
vit          95    74         14        100%       1             9 (全为权重)

- Q 与"激活 DQ"严格 1:1; Q 的输出 100% 只喂 DQ; 每族恰有 1 个 DQ 输出直连
  图输出; cnn 有 1 个 Q 的 float 输入直连图输入。
- Q.in_shared 的 4/9/9 条**全部是权重/偏置的再量化链**, 另一消费者是量化
  算子, 不构成"真实 float 读取", 故统一规则把它们正确归到 int8 侧。
- 真实 fp32 岛入口 (DQ 输出被 fp32 算子消费) 占比: gnn 5.4% / transformer
  6.8% / vit 8.5% / cnn 0% / rnn 0%; 岛内消费者固定为 LayerNorm 内部的
  Slice/ReduceMean/Sub/Sqrt, gnn 另加 BatchNormalization/Div/Expand/LSTM。
- dtype 声明链路: ORT 原始产物 native INT8 value_info = 0; 形状推理/onnxsim
  之后 Q 输出的 INT8 声明才出现 (cnn 125 / transformer 651), 故本 pass 必须
  挂在 simplifier **之后**。

遗留风险 (未完全消除, 已通过 qdq_boundary / qdq_residual 告警)
------------------------------------------------------------
R1 fp32 岛: 不在归一化区域内的算子按 fp32 处理。纯结构规则下, 若某个真
   fp32 岛恰好夹在真 DQ 与真 Q 之间 (其数据流本就从 DQ 走向 Q), 该岛会被
   并入区域、按 int8 计访存 (口径偏小)。区域内外仅由张量连接关系决定,
   无法进一步区分"这段究竟是真 int8 还是真 fp32", 属固有取舍。
R2 混合消费者: 同一 DQ 输出同时被量化算子与 fp32 算子消费时保留该 DQ, 量化
   侧会按 fp32 口径计访存 (偏大)。五族实测 mixed=0, 结构上仍可能发生。
R3 QOperator: QLinearConv/QGemm/MatMulInteger/ConvInteger 等自带量化语义、
   图内无 Q/DQ 的算子不在本 pass 覆盖范围, 仍走 unknown_op 告警。
R4 图 I/O 上的 Q/DQ: 保留为 Cast, 该张量按显式位宽计 (int8 侧 1B / float
   侧 4B), 与真实"入口即量化/出口即反量化"的执行一致, 但引入一次额外拷贝。
"""

from __future__ import annotations

import onnx

from ..diagnostics import issue


# 量化图表示用到的两个"类型转换"算子
_QDQ_OPS = ("QuantizeLinear", "DequantizeLinear")

# int8 量化张量的元素类型
_QUANT_TYPES = (onnx.TensorProto.INT8, onnx.TensorProto.UINT8)

# QOperator 风格量化算子: 自带量化语义、图内不含 Q/DQ, 本 pass 不处理, 仅告警
_QOPERATOR_OPS = (
    "QLinearConv",
    "QLinearMatMul",
    "QLinearAdd",
    "QLinearMul",
    "QLinearConcat",
    "QLinearAveragePool",
    "QLinearGlobalAveragePool",
    "QGemm",
    "MatMulInteger",
    "ConvInteger",
    "DynamicQuantizeLinear",
)


def _collect_types(graph):
    """张量名 -> ONNX elem_type (来自 initializer + 图输入/输出 + value_info)。"""
    types = {init.name: init.data_type for init in graph.initializer}
    for src in (graph.input, graph.value_info, graph.output):
        for item in src:
            if item.type.HasField("tensor_type"):
                types[item.name] = item.type.tensor_type.elem_type
    return types


def _build_adjacency(nodes):
    """张量名 -> 生产节点下标 / 消费节点下标列表。"""
    producer, consumers = {}, {}
    for idx, node in enumerate(nodes):
        for name in node.output:
            if name:
                producer[name] = idx
        for name in node.input:
            if name:
                consumers.setdefault(name, []).append(idx)
    return producer, consumers


def _make_cast(node, src, dst, to_type):
    """把 Q/DQ 节点就地改写为等价的 Cast (张量名不变)。"""
    node.op_type = "Cast"
    node.domain = ""

    del node.attribute[:]
    attr = node.attribute.add()
    attr.name = "to"
    attr.type = onnx.AttributeProto.INT
    attr.i = int(to_type)

    del node.input[:]
    node.input.append(src)
    del node.output[:]
    node.output.append(dst)


def normalize_qdq(model: onnx.ModelProto):
    """就地折叠 QDQ 量化图; 返回 (ModelProto, 是否发生归一化)。

    参数
    ----
        model: 已完成形状推理/简化的 ONNX ModelProto。

    返回
    ----
        (onnx.ModelProto, bool): 折叠后的模型 (就地修改) 与"是否疑似量化图"
        (即是否含 Q/DQ 节点)。非量化图直接原样返回, 零副作用, 天然幂等;
        调用方可据此决定是否打印归一化步骤。
    """
    graph = model.graph
    nodes = list(graph.node)

    # 非量化图: 不做任何事, 天然幂等
    if not any(node.op_type in _QDQ_OPS for node in nodes):
        return model, False

    graph_in = {item.name for item in graph.input}
    graph_out = {item.name for item in graph.output}
    initializers = {item.name for item in graph.initializer}
    types = _collect_types(graph)
    producer, consumers = _build_adjacency(nodes)

    # ---- 真 int8 量化张量: 图内已声明 INT8|UINT8 的张量, 加 Q 输出 ----
    # Q 的输出恒为 int8, 故 Q 一律视为真; DQ 的数据输入未被声明为 int8 时视为
    # 伪 DQ (如 tf2onnx 把 int32 zero_point / BN 常量包出的转换), 不参与区域。
    q_in, q_out = set(), set()
    int8_names = {name for name, t in types.items() if t in _QUANT_TYPES}
    for node in nodes:
        if node.op_type == "QuantizeLinear" and node.input and node.output:
            q_in.add(node.input[0])
            q_out.add(node.output[0])
    int8_names |= q_out

    real_dq = {
        idx
        for idx, node in enumerate(nodes)
        if node.op_type == "DequantizeLinear"
        and node.input
        and node.input[0] in int8_names
    }

    # ---- 归一化区域: 从"真 DQ 输出"前向可达 ∩ 从"真 Q 输入"后向可达 ----
    def _forward(seeds):
        hit, seen, stack = set(), set(seeds), list(seeds)
        while stack:
            for c in consumers.get(stack.pop(), ()):
                if c not in hit:
                    hit.add(c)
                    for name in nodes[c].output:
                        if name and name not in seen:
                            seen.add(name)
                            stack.append(name)
        return hit

    def _backward(seeds):
        hit, seen, stack = set(), set(seeds), list(seeds)
        while stack:
            p = producer.get(stack.pop())
            if p is not None and p not in hit:
                hit.add(p)
                for name in nodes[p].input:
                    if name and name not in seen:
                        seen.add(name)
                        stack.append(name)
        return hit

    entry = {name for idx in real_dq for name in nodes[idx].output}
    region = _forward(entry) & _backward(q_in)

    def is_barrier(idx):
        """既非 Q/DQ, 也不在归一化区域内 (真实按 fp32 读写的算子)。"""
        return nodes[idx].op_type not in _QDQ_OPS and idx not in region

    # ---- 折叠 ----
    alias = {}  # float 侧张量名 -> int8 侧规范名
    removed = set()  # 已折叠删除的节点下标
    boundary = []  # 保留的 dtype 转换边界 (节点名, 原因)
    residual = []  # 未覆盖 / 降级情形

    def resolve(name):
        seen = set()
        while name in alias and name not in seen:
            seen.add(name)
            name = alias[name]
        return name

    def needs_float(float_side, is_q):
        """float 侧张量是否必须真实保留为 float; 返回原因或 None。"""
        if is_q:
            if float_side in graph_in:
                return "图输入"
            if float_side in initializers:
                return "常量 float 张量"
            prod = producer.get(float_side)
            if prod is None:
                return "无生产者"
            if is_barrier(prod):
                return "fp32 生产算子"
        else:
            if float_side in graph_out:
                return "图输出"

        if any(is_barrier(c) for c in consumers.get(float_side, ())):
            return "fp32 消费算子"

        return None

    for idx, node in enumerate(nodes):
        if node.op_type not in _QDQ_OPS:
            continue

        src = node.input[0] if node.input else ""
        dst = node.output[0] if node.output else ""
        if not src or not dst:
            residual.append(f"{node.op_type} 缺输入/输出, 已跳过")
            continue

        is_q = node.op_type == "QuantizeLinear"
        float_side = src if is_q else dst
        quant_side = dst if is_q else src
        default_type = onnx.TensorProto.INT8 if is_q else onnx.TensorProto.FLOAT

        # 伪 DQ (量化侧非真 int8): 无法折叠, 仅等价改写为 Cast, 避免 DQ 未注册
        # 引发 unknown_op 告警。
        if not is_q and idx not in real_dq:
            _make_cast(node, src, dst, types.get(dst, default_type))
            boundary.append((node.op_type, dst, "非真 int8 量化侧"))
            continue

        if float_side in alias:
            # 共享 float 张量的再量化链: 该张量已被前序节点折叠为 int8 侧规范名,
            # 此处就地保留为 Cast (int8->int8 requantize) 以维持图完整。
            _make_cast(node, src, dst, types.get(dst, default_type))
            residual.append(f"{node.op_type}({dst}) 浮点侧与其它节点共享, 保留为 Cast")
            continue

        reason = needs_float(float_side, is_q)
        if reason is not None:
            _make_cast(node, src, dst, types.get(dst, default_type))
            boundary.append((node.op_type, float_side, reason))
        else:
            alias[float_side] = resolve(quant_side)
            removed.add(idx)

    # ---- 全图重写: 每个节点的每个输入/输出槽位走 resolve ----
    if alias:
        for idx, node in enumerate(nodes):
            if idx in removed:
                continue
            in_names = [resolve(x) if x else x for x in node.input]
            del node.input[:]
            node.input.extend(in_names)
            out_names = [resolve(x) if x else x for x in node.output]
            del node.output[:]
            node.output.extend(out_names)

        # 清理已别名失效的 value_info (避免遗留孤立张量声明)
        kept_vi = [vi for vi in graph.value_info if vi.name not in alias]
        if len(kept_vi) != len(graph.value_info):
            del graph.value_info[:]
            graph.value_info.extend(kept_vi)

    # ---- 删除已折叠的 Q/DQ 节点 ----
    if removed:
        del graph.node[:]
        graph.node.extend(n for i, n in enumerate(nodes) if i not in removed)

    # ---- 遗留风险告警 ----
    if boundary:
        reasons = {}
        for _op_type, name, reason in boundary:
            reasons.setdefault(reason, []).append(name)
        detail = "; ".join(f"{r}:{len(v)}" for r, v in reasons.items())
        issue(
            "qdq_boundary",
            f"保留 {len(boundary)} 处 dtype 转换边界 (已改写为 Cast, 语义等价): {detail}",
            count=len(boundary),
            reasons={r: len(v) for r, v in reasons.items()},
        )

    qoperator = sorted({n.op_type for n in nodes if n.op_type in _QOPERATOR_OPS})
    if qoperator:
        residual.append(f"存在 QOperator 风格量化算子 {qoperator}, 未归一化")

    if residual:
        issue(
            "qdq_residual",
            "量化图归一化存在未完全覆盖的情形, 访存口径可能偏离真实执行: "
            + " | ".join(residual[:8])
            + (" ..." if len(residual) > 8 else ""),
            count=len(residual),
        )

    return model, True
