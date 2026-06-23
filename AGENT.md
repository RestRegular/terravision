# AGENT.md — R 角色工作指南

## 基于EuroSAT遥感图像的多标签土地类型分类

> 本文档为 **R 角色（数据、特征工程 & 融合实验）** 的工作执行规范，供 AI Agent 自动化执行任务时遵循。

---

## 一、角色定位

| 项目 | 内容 |
|------|------|
| 角色代号 | **R** |
| 负责模块 | 数据获取、特征工程、降维、特征融合 |
| 技术链长度 | 全流程最长（贯穿阶段一~阶段三） |
| 预计工时 | ~14h |
| 主要难点 | 多标签构造、特征工程、融合实验 |

---

## 二、职责范围

### 阶段一：数据准备（R 主导，W 可并行）

1. 从 HuggingFace `datasets` 库下载 EuroSAT 数据集
2. 完成数据探索分析（EDA）
   - 统计各类别样本数量
   - 可视化样本图像
   - 检查图像尺寸、通道、像素分布
3. 按语义规则构造多标签体系，生成标签矩阵
4. 划分训练 / 验证 / 测试集（8:1:1，分层抽样）

### 阶段二：特征工程

1. 提取手工特征：
   - **颜色统计特征**：RGB各通道均值/标准差/直方图（~96维）
   - **纹理特征**：LBP（局部二值模式，~256维）
   - **梯度特征**：HOG（方向梯度直方图，~1764维）
   - **频域特征**：灰度图FFT幅度统计（~32维）
2. 拼接为统一特征矩阵并进行 **L2 归一化**（总维度约 2148）
3. **PCA 降维**至 128 维
4. 生成 **t-SNE 可视化**图

### 阶段三：融合实验（R 主导）

1. **早融合（Early Fusion）**：手工特征 + ResNet深度特征拼接后训练分类器
2. **晚融合（Late Fusion）**：各模型预测结果加权投票
3. 汇总 `ml_results.csv` + `dl_results.csv` + 融合结果，生成 `all_results.csv`

---

## 三、多标签构造规则

原始 EuroSAT 为单标签 10 类，通过语义规则扩展为多标签：

| 主标签 | 附加标签 | 规则依据 |
|--------|----------|----------|
| AnnualCrop | Vegetation, OpenLand | 农田具有植被与开阔地特征 |
| Forest | Vegetation, NaturalArea | 森林属于自然植被区域 |
| HerbaceousVegetation | Vegetation, OpenLand | 草地属于开阔植被 |
| Highway | Artificial, Transportation | 公路为人工交通设施 |
| Industrial | Artificial, Urban | 工业区属于城市人工区域 |
| Pasture | Vegetation, OpenLand | 牧场兼具植被与开阔地特征 |
| PermanentCrop | Vegetation, OpenLand | 与一年生农田类似 |
| Residential | Artificial, Urban | 居民区属于城市人工区域 |
| River | Water, NaturalArea | 河流属于自然水体 |
| SeaLake | Water, NaturalArea | 海湖属于自然水体 |

**辅助标签集合**：`{Vegetation, OpenLand, Artificial, Urban, Transportation, Water, NaturalArea}`

**最终标签维度**：17（10 主标签 + 7 辅助标签）

---

## 四、交付物清单

### 代码文件

| 文件 | 功能 |
|------|------|
| `data_prepare.py` | 数据下载、EDA、多标签构造、数据集划分 |
| `feature_extract.py` | 手工特征提取（HOG/LBP/颜色/FFT）+ L2归一化 |
| `feature_reduce.py` | PCA降维 + t-SNE可视化 |
| `fusion_early.py` | 早融合实验（手工特征 + 深度特征） |
| `fusion_late.py` | 晚融合实验（加权投票） |

### 数据文件

| 文件 | shape | 说明 |
|------|-------|------|
| `features/features_train.npy` | (N_train, 128) | PCA降维后的训练特征 |
| `features/features_test.npy` | (N_test, 128) | PCA降维后的测试特征 |
| `data/labels_train.npy` | (N_train, 17) | 多标签训练标签矩阵 |
| `data/labels_test.npy` | (N_test, 17) | 多标签测试标签矩阵 |
| `results/all_results.csv` | — | 汇总所有模型（含融合）指标 |

### 可视化文件

| 文件 | 说明 |
|------|------|
| `figures/tsne_features.png` | t-SNE 特征分布图 |

### 报告章节

- 数据集介绍
- 特征工程
- 融合实验

---

## 五、接口约定

### R 对外输出

| 接收方 | 文件 | 格式 |
|--------|------|------|
| L | `features_train.npy` / `features_test.npy` | shape: (N, 128) |
| 全员 | `labels_train.npy` / `labels_test.npy` | shape: (N, 17)，17=10主+7辅 |
| H | `all_results.csv` | 汇总所有模型（含融合）完整指标 |

### R 接收输入

| 输出方 | 文件 | 格式 |
|--------|------|------|
| L | `ml_results.csv` | 列：model / hamming_loss / macro_f1 / micro_f1 / mAP |
| W | `dl_results.csv` | 同上格式 |
| W | `resnet_features_train.npy` / `resnet_features_test.npy` | shape: (N, 2048)，用于早融合 |
| L | `predictions/predictions_svm_test.npy` | shape: (N_test, 17)，晚融合预测概率 |
| L | `predictions/predictions_rf_test.npy` | shape: (N_test, 17)，晚融合预测概率 |
| L | `predictions/predictions_xgb_test.npy` | shape: (N_test, 17)，晚融合预测概率 |
| W | `predictions/predictions_cnn_test.npy` | shape: (N_test, 17)，晚融合预测概率 |
| W | `predictions/predictions_resnet_test.npy` | shape: (N_test, 17)，晚融合预测概率 |

> **注意**：晚融合需要各模型在测试集上的预测概率矩阵（sigmoid 输出），而非硬标签。L 和 W 需在阶段三补充交付。

---

## 六、目录结构

```
project/
├── data/               # R 负责：原始数据与标签
│   ├── eurosat_raw/    # 原始图像
│   ├── labels_train.npy
│   └── labels_test.npy
├── features/           # R & W 负责：导出的特征矩阵
│   ├── features_train.npy       # R 输出（128维）
│   ├── features_test.npy        # R 输出（128维）
│   ├── resnet_features_train.npy # W 输出（2048维）
│   └── resnet_features_test.npy  # W 输出（2048维）
├── models/             # L & W 负责
├── fusion/             # R 负责：融合实验脚本
│   ├── fusion_early.py
│   └── fusion_late.py
├── results/            # 所有 csv 结果统一存放
│   ├── ml_results.csv   # L 输出
│   ├── dl_results.csv   # W 输出
│   └── all_results.csv  # R 汇总输出
├── figures/            # H 负责
└── report/             # 各自报告章节，H 最终整合
```

---

## 七、技术规范

### 技术栈

```
语言：Python 3.10+
数据：datasets（HuggingFace）、numpy、Pillow
特征：scikit-image、opencv-python
传统ML：scikit-learn、xgboost
降维：scikit-learn（PCA）、umap-learn（可选）
可视化：matplotlib、seaborn
```

### 评估指标

多标签分类统一采用以下指标（供 `all_results.csv` 使用）：

| 指标 | 说明 |
|------|------|
| Subset Accuracy | 全部标签完全正确的样本比例 |
| Hamming Loss | 标签预测错误率（越小越好） |
| Macro F1 | 各标签F1的宏平均 |
| Micro F1 | 所有标签整体F1 |
| mAP | 各标签平均精度均值 |

### `all_results.csv` 格式规范

```csv
model,hamming_loss,macro_f1,micro_f1,mAP,subset_accuracy
SVM,0.0521,0.7832,0.8521,0.8901,0.6234
RandomForest,0.0487,0.8012,0.8678,0.9056,0.6512
XGBoost,0.0456,0.8156,0.8745,0.9123,0.6721
CNN,0.0389,0.8521,0.9012,0.9345,0.7234
ResNet50,0.0312,0.8823,0.9234,0.9512,0.7654
EarlyFusion,0.0298,0.8912,0.9312,0.9589,0.7812
LateFusion,0.0301,0.8898,0.9301,0.9578,0.7789
```

---

## 八、协作节点

| 阶段节点 | R 的事项 |
|----------|----------|
| 阶段一结束 | 通知全员数据集就绪；输出 `labels_train/test.npy` |
| 阶段二结束 | 通知 L & H 特征矩阵已导出；输出 `features_train/test.npy` |
| 阶段三结束 | 接收 L 的 `ml_results.csv` 和 W 的 `dl_results.csv` + 深度特征，完成融合实验，输出 `all_results.csv` 交给 H |

---

## 九、执行原则

1. **接口严格对齐**：所有输出文件的 shape、列名、路径必须严格符合约定，确保 L / W / H 可直接读取
2. **可复现性**：所有随机操作设置 `random_state=42`，数据划分、PCA、t-SNE 均需固定种子
3. **增量交付**：每个阶段完成后立即输出对应文件，不阻塞下游成员
4. **不越界**：R 不编写 L / W / H 负责的代码（如 ml_*.py、dl_*.py、compare_all.py），仅消费其输出
5. **虚拟环境**：运行 Python 命令前必须确认并激活虚拟环境
6. **路径规范**：使用相对路径或 `pathlib.Path`，确保跨平台兼容
7. **交付物压缩包**：每个阶段完成后生成交付物压缩包，便于分发与版本管理

---

## 十、交付物压缩包命名规则

每个阶段完成后，R 需将交付物打包为 zip 文件分发给相关成员：

### 命名格式

```
R-stage{n}-{分发对象}.zip
```

- `{n}`：阶段编号（1/2/3）
- `{分发对象}`：
  - **全员分发**：无后缀（如 `R-stage1.zip`）
  - **特定成员**：`-to-{成员代号}`（如 `R-stage1-to-L.zip`、`R-stage1-to-W.zip`）

### 各阶段交付物清单

| 阶段 | 压缩包名称 | 分发对象 | 包含文件 |
|------|------------|----------|----------|
| 阶段一 | `R-stage1.zip` | 全员 | `data/labels_*.npy`、`data/indices_*.npy`、`data/label_info.json`、`figures/eda_*.png` |
| 阶段二 | `R-stage2-to-L.zip` | L | `features/features_train.npy`、`features/features_test.npy` |
| 阶段二 | `R-stage2.zip` | 全员（可选） | 同上 + t-SNE 图 |
| 阶段三 | `R-stage3-to-H.zip` | H | `results/all_results.csv`、融合实验代码、可视化图表 |

### 压缩包结构

压缩包内保持原始目录结构，便于解压后直接使用：

```
R-stage1.zip
├── data/
│   ├── labels_train.npy
│   ├── labels_test.npy
│   └── ...
├── figures/
│   └── eda_*.png
└── ...
```

---

*文档版本：v1.1 | R 角色 Agent 工作指南*
