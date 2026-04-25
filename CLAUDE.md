# CLAUDE.md

此文件为在该代码库中工作的 Claude Code (claude.ai/code) 提供指导。

## 项目概述

NeuralOperator 是一个用于在 PyTorch 中学习神经算子的综合库。它实现了傅里叶神经算子 (FNO) 和张量化神经算子 (TNO)，用于学习函数空间之间的映射。该库支持分辨率不变性学习，使得训练好的算子可以应用于任何分辨率的数据。

## 代码架构

主要组件组织在 `neuralop` 包中：

- `models/`：包含 FNO、TFNO（张量化 FNO）、UNO、FNOGNO、GINO 的实现
- `layers/`：核心神经算子层，包括 SpectralConv、MLP 等
- `training/`：训练器类、回调函数和训练工具
- `data/`：数据处理和数据集工具
- `datasets/`：预定义数据集，如 Darcy 流
- `losses/`：专门用于算子学习的损失函数（LpLoss、H1Loss 等）
- `mpu/`：模型并行工具
- `tests/`：所有组件的单元测试

## 常见开发任务

### 安装
```bash
# 以开发模式安装
pip install -e .
pip install -r requirements.txt

# 安装开发依赖
pip install -r requirements_dev.txt
```

### 运行测试
```bash
# 运行所有测试
pytest -v neuralop

# 运行特定测试文件
pytest -v neuralop/tests/test_utils.py
```

### 代码格式化
```bash
# 使用 black 格式化代码
black .
```

### 构建文档
```bash
# 构建文档
cd doc
make html

# 本地查看文档
cd doc/build/html
python -m http.server 8000
```

## 关键类和 API

1. **模型**：`TFNO`、`FNO`、`UNO` - 主要的神经算子模型
2. **训练器**：`Trainer` 类用于训练神经算子
3. **损失函数**：`LpLoss`、`H1Loss` 用于函数空间损失
4. **数据**：`load_darcy_flow_small` 和其他数据集加载器

## 使用示例模式

```python
from neuralop.models import TFNO
from neuralop import Trainer
from neuralop.data.datasets import load_darcy_flow_small
from neuralop import LpLoss, H1Loss

# 加载数据
train_loader, test_loaders = load_darcy_flow_small(...)

# 创建模型
model = TFNO(n_modes=(16, 16), hidden_channels=32, factorization='tucker')

# 设置训练
optimizer = torch.optim.Adam(model.parameters(), lr=8e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
loss = H1Loss(d=2)

# 训练
trainer = Trainer(model=model, n_epochs=20)
trainer.train(train_loader, test_loaders, optimizer, scheduler, loss)
```