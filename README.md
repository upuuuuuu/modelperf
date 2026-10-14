# ModelPerf — A Deep Learning Model Performance Analyzer

ModelPerf turns any deep-learning model into a **computation graph** annotated with
operator-level cost, and lets you explore it in the browser — **without ever running the
model**.

- **Any framework**: PyTorch · TensorFlow · Keras · TFLite · ONNX · PaddlePaddle · Caffe · JAX · MindSpore · OneFlow · Chainer · MXNet, plus `seer_ir`, and more
- **Any model**: CNN · GNN · RNN / LSTM / GRU · Transformer / ViT · MLP — 100+ operators
- **Whole graph**: layer count, operator mix, topology & connectivity, overall cost
- **Each node (op)**: operator type, input/output shapes, dtype, compute (FLOPs), memory access (MAC / bytes), time cost
- **Each edge (tensor)**: shape, dtype, byte size, node-to-node data flow
- **Interactive web UI**: zoom, search, and navigate the full graph, nodes, and edges in the browser
- **Fully static analysis**: no model execution, no real data, no GPU required
- **Optional multi-hardware targets**: two cost models — roofline (upper bound) and serial (compute + memory)

## Quick start

Install ModelPerf together with the framework you want to analyze — conversion goes
through ONNX:

```bash
pip install modelperf            # base package (ONNX / seer_ir inputs)
pip install "modelperf[torch]"     # + PyTorch backend
pip install "modelperf[tensorflow]"  # + TensorFlow / Keras / TFLite backend
```

```python
import modelperf

# model -> ONNX -> IR -> computation graph, with roofline cost injected per device
ir, cg = modelperf.build_computation_graph(
    model=model,            # a model object, or a path for file / directory based formats
    model_format="torch",   # see modelperf.validation.SUPPORTED_MODEL_FORMATS
    data_dict=data,         # {input_name: {"shape": (...), "dtype": "..."}}
    devices=["rtx3090"],    # target hardware; omit to skip cost injection
)

modelperf.report(cg)                      # terminal statistics
modelperf.export_html(cg, "graph.html")   # self-contained interactive HTML
modelperf.visualize_graph(cg)             # local interactive viewer
```

Run the bundled example:

```bash
python -m modelperf.demo
```
