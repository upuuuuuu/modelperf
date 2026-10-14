"""各框架/格式 -> ONNX 的统一转换入口 (any2onnx)。

本模块同时承担「模型加载 / 路径解析」职责 (原 model_loader 的职能已内联于此)：
`model` 既可以是已加载的对象, 也可以是文件/目录路径。

key 约定:
  - torch 单 key: 已加载 ``nn.Module`` 与 TorchScript(``.pt``/``.pth``) 同走
    ``torch.onnx.export``, 均需 data_dict 提供示例输入。
  - tensorflow 族四 key: ``keras`` / ``tensorflow``(等价 Keras) / ``tf_savedmodel`` /
    ``tf_graphdef`` / ``tflite``。四者格式与后端 API 不同, 对外用独立 key,
    内部 ``key -> 固定管道``, 不做文件后缀嗅探。

所有第三方后端均在函数内惰性导入, 仅在对应格式被调用时才需要依赖。
"""

from __future__ import annotations

import os
import tempfile
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import onnx

from ..diagnostics import issue


# --------------------------------------------------------------------------- #
# 路径解析辅助 (原 model_loader 职责)
# --------------------------------------------------------------------------- #
def _is_path(x: Any) -> bool:
    return isinstance(x, (str, os.PathLike))


def _find_one(pattern: str) -> Optional[str]:
    matches = glob(pattern)
    return matches[0] if matches else None


def _as_path(x: Any, what: str) -> str:
    if not _is_path(x):
        issue(
            "invalid_input",
            f"{what} 需为路径 (str/os.PathLike), 收到: {type(x).__name__}",
        )
    p = os.fspath(x)
    if not os.path.exists(p):
        issue("invalid_input", f"{what} 不存在: {p}")
    return p


def _load_onnx(model: Any) -> onnx.ModelProto:
    return onnx.load(_as_path(model, "ONNX 模型")) if _is_path(model) else model


def _load_mxnet(model: Any) -> Tuple[str, str]:
    """解析 MXNet 符号图与参数文件: (sym_json, params_path)。"""
    if not _is_path(model):
        issue("invalid_input", "MXNet 需提供路径 (目录或 *-symbol.json)")
    path = os.fspath(model)
    if os.path.isdir(path):
        sym = _find_one(os.path.join(path, "*-symbol.json"))
        params = _find_one(os.path.join(path, "*.params"))
    elif path.endswith("-symbol.json"):
        sym = path
        params = _find_one(path.replace("-symbol.json", "-*.params"))
    else:
        sym = params = None
    if not (sym and params):
        issue("invalid_input", f"未找到 MXNet symbol/params 文件: {path}")
    return sym, params


def _load_paddle(model: Any) -> Tuple[str, str, Optional[str]]:
    """解析 Paddle 推理模型目录: (model_dir, model_filename, params_filename)。

    Paddle 2.x 导出 ``inference.pdmodel``, Paddle 3.x 导出 ``inference.json``,
    故按实际存在的文件探测; 参数文件缺失时返回 ``None``。
    """
    if not _is_path(model):
        issue("invalid_input", "Paddle 需提供模型目录路径")
    d = os.fspath(model)
    if not os.path.isdir(d):
        issue("invalid_input", f"Paddle 模型目录不存在: {d}")
    model_file = _find_one(os.path.join(d, "*.pdmodel")) or _find_one(
        os.path.join(d, "*.json")
    )
    if not model_file:
        issue("invalid_input", f"未找到 Paddle 模型文件 (*.pdmodel/*.json): {d}")
    params_file = _find_one(os.path.join(d, "*.pdiparams"))
    return (
        d,
        os.path.basename(model_file),
        (os.path.basename(params_file) if params_file else None),
    )


def _load_caffe(model: Any) -> Tuple[str, str]:
    """解析 Caffe 模型目录: (prototxt, caffemodel)。"""
    if not _is_path(model):
        issue("invalid_input", "Caffe 需提供模型目录路径")
    d = os.fspath(model)
    if not os.path.isdir(d):
        issue("invalid_input", f"Caffe 模型目录不存在: {d}")
    prototxt = _find_one(os.path.join(d, "*.prototxt"))
    caffemodel = _find_one(os.path.join(d, "*.caffemodel"))
    if not (prototxt and caffemodel):
        issue("invalid_input", f"未找到 Caffe prototxt/caffemodel: {d}")
    return prototxt, caffemodel


# --------------------------------------------------------------------------- #
# PyTorch (单 key: nn.Module / TorchScript)
# --------------------------------------------------------------------------- #
def torch_export(
    model: Any, *, data_dict: Dict[str, Any], opset: int = 16
) -> onnx.ModelProto:
    import torch

    if _is_path(model):
        model = torch.jit.load(os.fspath(model), map_location="cpu")
    model.eval()

    # TorchScript (ScriptModule/ScriptFunction) 不接受 dict/kwargs 形式的示例输入,
    # 必须按位置传参; eager nn.Module 则用 dict 以便按 key 命名输入。
    args = (
        tuple(data_dict.values())
        if isinstance(model, torch.jit.ScriptModule)
        else data_dict
    )

    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    tmp.close()
    try:
        torch.onnx.export(
            model,
            args,
            tmp.name,
            opset_version=opset,
            do_constant_folding=False,
        )
        return onnx.load(tmp.name)
    finally:
        os.unlink(tmp.name)


# --------------------------------------------------------------------------- #
# TensorFlow / Keras
# --------------------------------------------------------------------------- #
def keras_export(
    model: Any, *, tf_signature: List[Any], opset: int = 16
) -> onnx.ModelProto:
    import tensorflow as tf
    import tf2onnx

    if _is_path(model):
        model = tf.keras.models.load_model(os.fspath(model))
    onnx_model, _ = tf2onnx.convert.from_keras(
        model, input_signature=tf_signature, opset=opset, output_path=None
    )
    return onnx_model


def tf_savedmodel_export(saved_model_dir: Any, *, opset: int = 16) -> onnx.ModelProto:
    import tf2onnx
    from tf2onnx import tf_loader

    path = _as_path(saved_model_dir, "SavedModel 目录")
    # tf2onnx 自带的 SavedModel 载入器会冻结变量、过滤 captured inputs, 并把
    # serving 签名名映射回结构化输入/输出名 (tensors_to_rename)。
    graph_def, inputs, outputs, _initialized_tables, tensors_to_rename = (
        tf_loader.from_saved_model(
            path,
            None,
            None,
            None,
            None,
            None,
            False,
            return_initialized_tables=True,
            return_tensors_to_rename=True,
        )
    )
    onnx_model, _ = tf2onnx.convert.from_graph_def(
        graph_def,
        input_names=inputs,
        output_names=outputs,
        opset=opset,
        tensors_to_rename=tensors_to_rename,
        output_path=None,
    )
    return onnx_model


def _graphdef_io_names(graph_def: Any) -> Tuple[List[str], List[str]]:
    """从 GraphDef 推断 ``from_graph_def`` 所需的输入/输出名。

    tf2onnx 期望 ``node_name:port_id`` 形式, 故统一补 ``:0``。
    - 输入: ``Placeholder`` / ``PlaceholderWithDefault`` 节点
    - 输出: 输出未被任何其它节点消费的终端节点 (排除源节点与 NoOp 控制节点)
    """
    # 源节点没有消费者, 但它们不是图输出, 需显式排除; NoOp 只承载控制依赖。
    _sources = ("Placeholder", "PlaceholderWithDefault", "Const", "NoOp")
    consumed = set()
    for node in graph_def.node:
        for inp in node.input:
            name = inp[1:] if inp.startswith("^") else inp  # 去掉控制依赖前缀
            consumed.add(name.split(":")[0])
    input_names = [
        f"{n.name}:0"
        for n in graph_def.node
        if n.op in ("Placeholder", "PlaceholderWithDefault")
    ]
    output_names = [
        f"{n.name}:0"
        for n in graph_def.node
        if n.name not in consumed and n.op not in _sources
    ]
    return input_names, output_names


def tf_graphdef_export(graph_def_path: Any, *, opset: int = 16) -> onnx.ModelProto:
    import tensorflow as tf
    import tf2onnx

    path = _as_path(graph_def_path, "GraphDef 文件")
    graph_def = tf.compat.v1.GraphDef()
    with open(path, "rb") as f:
        graph_def.ParseFromString(f.read())

    input_names, output_names = _graphdef_io_names(graph_def)
    if not input_names or not output_names:
        issue(
            "invalid_input",
            f"无法从 GraphDef 推断输入/输出节点: inputs={input_names}, outputs={output_names}",
        )
    onnx_model, _ = tf2onnx.convert.from_graph_def(
        graph_def,
        name=os.path.basename(path),
        input_names=input_names,
        output_names=output_names,
        opset=opset,
        output_path=None,
    )
    return onnx_model


def tflite_export(tflite_path: Any, *, opset: int = 16) -> onnx.ModelProto:
    import tf2onnx

    path = _as_path(tflite_path, "TFLite 文件")
    onnx_model, _ = tf2onnx.convert.from_tflite(path, opset=opset, output_path=None)
    return onnx_model


# --------------------------------------------------------------------------- #
# MXNet / Paddle / Caffe / Chainer
# --------------------------------------------------------------------------- #
def mxnet_export(
    sym_json: str,
    params_path: str,
    *,
    input_shapes: List[Tuple],
) -> onnx.ModelProto:
    import mxnet.onnx as mx_onnx

    # mxnet.onnx.export_model 只接受文件路径 (无 output_path=None 选项), 故落临时
    # 目录再读回; 用 TemporaryDirectory 保证退出即清理且无 mktemp 竞态。
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "model.onnx")
        # 输入名取自 symbol 图本身, 不支持 input_names/opset_version
        mx_onnx.export_model(
            sym_json,
            params_path,
            in_shapes=input_shapes,
            onnx_file_path=out,
        )
        return onnx.load(out)


def paddle_export(
    model_dir: str,
    model_filename: str,
    params_filename: Optional[str],
    *,
    opset: int = 16,
) -> onnx.ModelProto:
    import paddle2onnx

    onnx_bytes = paddle2onnx.export(
        model_filename=os.path.join(model_dir, model_filename),
        params_filename=(
            os.path.join(model_dir, params_filename) if params_filename else None
        ),
        opset_version=opset,
        enable_onnx_checker=False,
    )
    return onnx.load_from_string(bytes(onnx_bytes))


def caffe_export(prototxt: str, caffemodel: str) -> onnx.ModelProto:
    from caffe2onnx.convert import Caffe2Onnx, loadcaffemodel  # type: ignore

    graph, params = loadcaffemodel(prototxt, caffemodel)
    return Caffe2Onnx(graph, params, "caffe_model").createOnnxModel()


def chainer_export(
    chainer_model: Any, *, numpy_inputs: List[Any], opset: int = 16
) -> onnx.ModelProto:
    import onnx_chainer  # type: ignore

    # onnx_chainer.export 只接受文件路径, 落临时目录再读回并自动清理。
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "model.onnx")
        x = numpy_inputs if len(numpy_inputs) > 1 else numpy_inputs[0]
        onnx_chainer.export(chainer_model, x, filename=out, opset_version=opset)
        return onnx.load(out)


# --------------------------------------------------------------------------- #
# JAX
# --------------------------------------------------------------------------- #
def jax_export(model: Any, *, inputs: List[Any], opset: int = 16) -> onnx.ModelProto:
    """JAX 函数 -> ONNX (jax2onnx 直接导出, 不经过 TF)。"""
    import jax2onnx

    return jax2onnx.to_onnx(model, inputs, model_name="jax_model", opset=opset)


# --------------------------------------------------------------------------- #
# MindSpore
# --------------------------------------------------------------------------- #
def mindspore_export(model: Any, *, inputs: List[Any]) -> onnx.ModelProto:
    """MindSpore 网络 -> ONNX。

    优先用非弃用的 ``mindspore.onnx.export``; 老版本回退到
    ``mindspore.export(file_format='ONNX')``。导出结果始终为静态形状。
    """
    import mindspore as ms

    # ms.export 只接受文件路径, 落临时目录再读回并自动清理。
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "model.onnx")
        if hasattr(ms, "onnx"):
            ms.onnx.export(model, *inputs, file_name=out)
        else:
            ms.export(model, *inputs, file_name=out, file_format="ONNX")
        return onnx.load(out)


# --------------------------------------------------------------------------- #
# OneFlow
# --------------------------------------------------------------------------- #
def oneflow_export(
    model: Any, *, inputs: List[Any], opset: Optional[int] = None
) -> onnx.ModelProto:
    """OneFlow 静态图 -> ONNX (oneflow-onnx / flow2onnx)。

    oneflow-onnx 只接受已编译的 ``nn.Graph`` (依赖 ``graph._full_graph_proto``),
    故先把 nn.Module 包一层 Graph 并用示例输入触发编译。
    """
    from oneflow_onnx.oneflow2onnx.util import export_onnx_model

    graph = _as_oneflow_graph(model, inputs)
    work_dir = tempfile.mkdtemp()
    onnx_path, cleanup = export_onnx_model(
        graph,
        opset=opset,
        onnx_model_path=work_dir,
        dynamic_batch_size=False,
    )
    try:
        return onnx.load(onnx_path)
    finally:
        cleanup()


def _as_oneflow_graph(model: Any, inputs: List[Any]) -> Any:
    """把 nn.Module 包装成 nn.Graph 并用示例输入编译; 已是 Graph 则直接复用。"""
    import oneflow as flow

    if isinstance(model, flow.nn.Graph):
        graph = model
    else:
        module = model

        class _Graph(flow.nn.Graph):
            def __init__(self) -> None:
                super().__init__()
                self.module = module

            def build(self, *args: Any) -> Any:
                return self.module(*args)

        graph = _Graph()

    # _full_graph_proto 仅在首次编译后生成
    graph._compile(*inputs)
    return graph


# --------------------------------------------------------------------------- #
# 统一分派
# --------------------------------------------------------------------------- #
def onnx_converter(
    model: Any,
    model_format: str,
    data_dict: Any,
    opset: int = 16,
) -> onnx.ModelProto:
    fmt = str(model_format).lower()

    if fmt == "onnx":
        return _load_onnx(model)
    if fmt == "torch":
        return torch_export(model, data_dict=data_dict, opset=opset)
    if fmt in ("tensorflow", "keras"):
        return keras_export(model, tf_signature=data_dict, opset=opset)
    if fmt == "tf_savedmodel":
        return tf_savedmodel_export(model, opset=opset)
    if fmt == "tf_graphdef":
        return tf_graphdef_export(model, opset=opset)
    if fmt == "tflite":
        return tflite_export(model, opset=opset)
    if fmt == "mxnet":
        sym_json, params_path = _load_mxnet(model)
        input_shapes, _ = data_dict
        return mxnet_export(
            sym_json,
            params_path,
            input_shapes=input_shapes,
        )
    if fmt == "paddle":
        model_dir, model_filename, params_filename = _load_paddle(model)
        return paddle_export(model_dir, model_filename, params_filename, opset=opset)
    if fmt == "chainer":
        return chainer_export(model, numpy_inputs=data_dict, opset=opset)
    if fmt == "caffe":
        prototxt, caffemodel = _load_caffe(model)
        return caffe_export(prototxt, caffemodel)
    if fmt == "jax":
        return jax_export(model, inputs=data_dict, opset=opset)
    if fmt == "mindspore":
        return mindspore_export(model, inputs=data_dict)
    if fmt == "oneflow":
        return oneflow_export(model, inputs=data_dict, opset=opset)

    issue("invalid_input", f"Unknown model_format: {model_format}")
