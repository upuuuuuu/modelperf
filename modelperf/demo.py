"""最小示例：用 ModelPerf 分析一个官方 CNN 模型的计算图。

运行方式::

    python -m modelperf.demo

走完整流水线：模型 -> ONNX -> IR -> CG，并按目标硬件注入 roofline 代价。
"""

from torchvision.models import resnet50

from .analysis import report
from .cg_builder import build_computation_graph


def main():
    # 官方 ResNet-50; weights=None 表示不加载预训练权重 (免联网下载)
    model = resnet50(weights=None).eval()

    # 规范形状字典: 输入名 -> {shape, dtype}
    # 输入名 "x" 必须与 model.forward 的形参名一致; ModelPerf 据此内部造示例张量
    data_dict = {
        "x": {"shape": (1, 3, 224, 224), "dtype": "float32"},
    }

    # 1. 构图: 模型 -> ONNX -> IR -> CG, 并按设备注入 roofline 代价
    ir, cg = build_computation_graph(
        model=model,  # 模型对象 (onnx/tf_savedmodel 等为文件/目录路径)
        model_format="torch",  # 输入模型格式: torch/onnx/tensorflow/keras/... (见 validation)
        data_dict=data_dict,  # 规范形状字典, 用于造示例张量并导出 ONNX
        batch_size=1,  # 批大小元数据 (仅图级展示用, 不影响任何指标; 真实 batch 由图/形状字典决定)
        devices=["rtx3090"],  # 目标硬件列表, 决定注入哪套 roofline 代价
        optimize=True,  # 简化时是否顺带做图优化
        opset=16,  # 导出 ONNX 使用的 opset 版本
    )
    print(f"IR: {ir.number_of_nodes()} nodes, {ir.number_of_edges()} edges")
    print(f"CG: {cg.number_of_nodes()} nodes, {cg.number_of_edges()} edges")

    # 2. 分析 (及可选可视化): 默认只打印终端统计
    report(
        cg,  # 待分析的计算图 (节点/边带 feature 属性)
        detailed=False,  # True 时额外输出计算 / 内存 / 形状键
        csv_path=None,  # 传入路径则把节点级 DataFrame 导出为 CSV
        show_viewer=True,  # True 则启动交互式 Web viewer
        port=0,  # viewer 端口, 0 = 自动选择
    )


if __name__ == "__main__":
    main()
