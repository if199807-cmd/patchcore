import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import torch
torch.set_float32_matmul_precision('medium')
# 可选 high，速度更快，精度损失稍微大一点
# torch.set_float32_matmul_precision('high')


from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine


def main():
    print("CUDA available:", torch.cuda.is_available())

    # 1. 数据集
    datamodule = MVTecAD(
        root="./MVTecAD",
        category="bottle",
        train_batch_size=4,
        eval_batch_size=4,
        num_workers=4,  # Windows 先用 0，确保正常运行
    )

    # 2. PatchCore 模型
    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
    )

    # 3. 训练引擎
    engine = Engine(
        accelerator="gpu",
        devices=1,
        max_epochs=1,
        default_root_dir="./results",
    )

    # 4. 构建正常特征库
    engine.fit(
        model=model,
        datamodule=datamodule,
    )

    # 5. 测试模型
    results = engine.test(
        model=model,
        datamodule=datamodule,
    )

    print("Test results:", results)


if __name__ == "__main__":
    main()
