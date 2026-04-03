# addmoudle: 基于熵的多尺度特征融合模块

本分支实现了一个模块化的熵驱动尺度加权机制，用于在 **cost aggregation 后的 cost volume** 上估计类别尺度倾向，并对中间层特征进行加权融合。

## 设计目标

- 在聚合后的 cost volume 上计算类别空间熵，而非原始 cost volume。
- 对类别熵做归一化排名，得到 soft scale score。
- 使用 MLP 将 soft scale score 映射为多尺度融合权重。
- 对低置信度类别使用均匀权重回退，避免“类别不存在时熵虚高”。
- 保持模块化设计，便于后续替换策略。

## 代码结构

- 新增模块：
  - `/home/runner/work/ovrs-b/ovrs-b/cat_seg/modeling/transformer/entropy_scale_fusion.py`
    - `EntropyScaleFusion`：负责熵计算、排名归一化、MLP权重映射、置信度回退。
- 集成位置：
  - `/home/runner/work/ovrs-b/ovrs-b/cat_seg/modeling/transformer/model_ours.py`
    - `Aggregator` 中在 cost aggregation 后（`corr_embed`）调用 `EntropyScaleFusion`。
    - 新增 `fuse_guidance` 用于对 `res3/res4/res5` 进行按类别加权融合。
  - `/home/runner/work/ovrs-b/ovrs-b/cat_seg/modeling/transformer/cat_seg_predictor_ours.py`
    - 新增配置透传到 `Aggregator`。
  - `/home/runner/work/ovrs-b/ovrs-b/cat_seg/config.py` 与 `/home/runner/work/ovrs-b/ovrs-b/configs/config.yaml`
    - 新增熵融合相关配置项。

## 熵与权重计算流程

设 cost aggregation 后得到 `M`（按类别的2D响应图），对每个类别执行：

1. 空间概率归一化（温度系数 `tau`）  
2. 计算空间熵  
3. 归一化到 `(0, 1)`  
4. 在类别维做 rank 并归一化得到 `r_c`  
5. 将 `r_c` 输入 MLP，经过 softmax 输出 3 路尺度权重  
6. 若类别置信度低于阈值，回退为均匀权重 `(1/3, 1/3, 1/3)`

## 关键配置项

位于 `MODEL.SEM_SEG_HEAD`：

- `ENTROPY_TEMPERATURE`: softmax 温度
- `ENTROPY_EPS`: 数值稳定项
- `ENTROPY_CONFIDENCE_THRESHOLD`: 置信度阈值（低于阈值回退均匀权重）
- `ENTROPY_MLP_HIDDEN_DIM`: MLP隐层维度
- `ENTROPY_STAGE`: 两阶段训练开关（1或2）

## 两阶段训练建议

### 第一阶段（基础收敛）

- `ENTROPY_STAGE: 1`
- 行为：冻结 MLP，使用均匀权重进行融合
- 目的：先让基础分割能力、cost volume 和 skip connection 收敛

### 第二阶段（端到端学习）

- `ENTROPY_STAGE: 2`
- 行为：解冻 MLP，分割损失端到端反传到 MLP
- 目的：学习不同类别在不同尺度下的最优融合偏好

## 训练示例

阶段一：

```bash
python train_net.py --config configs/vitl_336.yaml MODEL.SEM_SEG_HEAD.ENTROPY_STAGE 1
```

阶段二（加载阶段一权重继续训练）：

```bash
python train_net.py --config configs/vitl_336.yaml MODEL.SEM_SEG_HEAD.ENTROPY_STAGE 2 MODEL.WEIGHTS <stage1_checkpoint>
```

