"""
data_prepare.py
R 角色 - 阶段一：数据准备

功能：
1. 从 HuggingFace 下载 EuroSAT 数据集
2. EDA 探索分析（类别分布、样本可视化、图像统计）
3. 按语义规则构造多标签矩阵（17维 = 10主 + 7辅）
4. 分层划分训练/验证/测试集（8:1:1）

交付物：
- data/labels_train.npy  (N_train, 17)
- data/labels_val.npy    (N_val, 17)
- data/labels_test.npy   (N_test, 17)
- data/indices_*.npy     划分索引（供 W/L 对齐原始图像与特征矩阵）
- data/label_info.json   标签映射信息
- figures/eda_*.png      EDA 可视化
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# ============================================================
# 配置
# ============================================================
RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# EuroSAT 在 HuggingFace 上的候选标识（按优先级尝试）
HF_DATASET_CANDIDATES = [
    "timm/eurosat-rgb",
    "blanchon/EuroSAT_RGB",
    "danjacobellis/EuroSAT",
]

# ============================================================
# 标签定义
# ============================================================
MAIN_LABELS = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]
AUX_LABELS = [
    "Vegetation", "OpenLand", "Artificial", "Urban",
    "Transportation", "Water", "NaturalArea",
]
ALL_LABELS = MAIN_LABELS + AUX_LABELS  # 17维

# 主标签 -> 辅助标签 映射规则（依据选题文档 2.3 节）
MULTI_LABEL_MAP = {
    "AnnualCrop":           ["Vegetation", "OpenLand"],
    "Forest":               ["Vegetation", "NaturalArea"],
    "HerbaceousVegetation": ["Vegetation", "OpenLand"],
    "Highway":              ["Artificial", "Transportation"],
    "Industrial":           ["Artificial", "Urban"],
    "Pasture":              ["Vegetation", "OpenLand"],
    "PermanentCrop":        ["Vegetation", "OpenLand"],
    "Residential":          ["Artificial", "Urban"],
    "River":                ["Water", "NaturalArea"],
    "SeaLake":              ["Water", "NaturalArea"],
}


# ============================================================
# 工具函数
# ============================================================
def build_multi_label(main_label_idx: int) -> np.ndarray:
    """根据主标签索引构建 17 维多标签向量（主标签 + 辅助标签）。"""
    main_name = MAIN_LABELS[main_label_idx]
    vec = np.zeros(len(ALL_LABELS), dtype=np.uint8)
    vec[main_label_idx] = 1  # 主标签
    for aux in MULTI_LABEL_MAP[main_name]:
        aux_idx = len(MAIN_LABELS) + AUX_LABELS.index(aux)
        vec[aux_idx] = 1
    return vec


def download_eurosat():
    """从 HuggingFace 下载 EuroSAT 数据集，依次尝试候选标识。

    timm/eurosat-rgb 预分了 train/validation/test 三 split（共 27000 张），
    本项目需要合并后按 8:1:1 重新划分，因此返回合并后的 Dataset 对象。
    """
    print("=" * 60)
    print("步骤 1: 下载 EuroSAT 数据集")
    print("=" * 60)
    last_err = None
    for name in HF_DATASET_CANDIDATES:
        try:
            print(f"  尝试加载: {name}")
            ds = load_dataset(name)
            print(f"  成功加载: {name}")
            print(f"  splits: {list(ds.keys())}")

            # 合并所有 split，得到 27000 张全集
            from datasets import concatenate_datasets
            merged = concatenate_datasets(list(ds.values()))
            print(f"  合并后总样本数: {len(merged)}")
            return merged
        except Exception as e:
            print(f"  失败: {e.__class__.__name__}: {str(e)[:120]}")
            last_err = e
    raise RuntimeError(f"所有候选数据集标识均加载失败: {last_err}")


def eda(train_ds):
    """探索性数据分析：类别分布、样本可视化、图像统计。

    参数 train_ds 实际为合并后的全集 Dataset。
    """
    print("\n" + "=" * 60)
    print("步骤 2: EDA 探索分析")
    print("=" * 60)

    main_labels = np.array(train_ds["label"])

    unique, counts = np.unique(main_labels, return_counts=True)
    print(f"\n总样本数: {len(train_ds)}")
    print(f"类别数: {len(unique)}")
    print("\n类别分布:")
    for u, c in zip(unique, counts):
        print(f"  {MAIN_LABELS[u]:25s}: {c:5d} ({c / len(train_ds) * 100:.1f}%)")

    # ---- 图1: 类别分布柱状图 ----
    plt.figure(figsize=(12, 5))
    ax = sns.barplot(
        x=[MAIN_LABELS[u] for u in unique],
        y=counts,
        color="steelblue",
    )
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("EuroSAT Class Distribution")
    plt.xticks(rotation=30, ha="right")
    for i, c in enumerate(counts):
        ax.text(i, c, str(c), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_class_distribution.png", dpi=150)
    plt.close()
    print(f"\n[图] 类别分布: figures/eda_class_distribution.png")

    # ---- 图2: 各类样本展示 ----
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    shown = set()
    for sample in train_ds:
        lbl = sample["label"]
        if lbl in shown:
            continue
        shown.add(lbl)
        ax = axes[lbl // 5, lbl % 5]
        ax.imshow(sample["image"])
        ax.set_title(MAIN_LABELS[lbl], fontsize=10)
        ax.axis("off")
        if len(shown) == 10:
            break
    plt.suptitle("EuroSAT Sample Images per Class", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_samples.png", dpi=150)
    plt.close()
    print(f"[图] 样本展示: figures/eda_samples.png")

    # ---- 图像基本统计 ----
    sample_img = np.array(train_ds[0]["image"])
    print(f"\n图像尺寸: {sample_img.shape}")
    print(f"像素值范围: [{sample_img.min()}, {sample_img.max()}]")
    print(f"数据类型: {sample_img.dtype}")

    return train_ds


def construct_multilabel(train_ds):
    """构造 17 维多标签矩阵。"""
    print("\n" + "=" * 60)
    print("步骤 3: 多标签构造")
    print("=" * 60)

    n = len(train_ds)
    main_labels = np.array(train_ds["label"])
    multi_labels = np.zeros((n, len(ALL_LABELS)), dtype=np.uint8)
    for i, lbl in enumerate(main_labels):
        multi_labels[i] = build_multi_label(lbl)

    print(f"多标签矩阵 shape: {multi_labels.shape}")
    print(f"每样本平均标签数: {multi_labels.sum(axis=1).mean():.2f}")
    print("\n各标签样本数:")
    for i, name in enumerate(ALL_LABELS):
        tag = "[主]" if i < len(MAIN_LABELS) else "[辅]"
        print(f"  {tag} {name:25s}: {int(multi_labels[:, i].sum()):5d}")

    return multi_labels


def split_dataset(multi_labels):
    """分层划分训练/验证/测试集 8:1:1，并保存索引与标签。"""
    print("\n" + "=" * 60)
    print("步骤 4: 数据集划分 (8:1:1)")
    print("=" * 60)

    n = len(multi_labels)
    indices = np.arange(n)

    # 用主标签作为分层依据（多标签无法直接分层，取主标签即可，因每样本必有且仅有一个主标签）
    main_label_col = multi_labels[:, :len(MAIN_LABELS)]
    stratify_key = main_label_col.argmax(axis=1)

    # 先分出 20% 作为 (验证+测试)
    train_idx, testval_idx = train_test_split(
        indices, test_size=0.2, random_state=RANDOM_STATE,
        shuffle=True, stratify=stratify_key,
    )
    # 再将 20% 平分为验证和测试各 10%
    val_idx, test_idx = train_test_split(
        testval_idx, test_size=0.5, random_state=RANDOM_STATE,
        shuffle=True, stratify=stratify_key[testval_idx],
    )

    train_labels = multi_labels[train_idx]
    val_labels = multi_labels[val_idx]
    test_labels = multi_labels[test_idx]

    print(f"训练集: {len(train_idx):5d} ({len(train_idx) / n * 100:.1f}%)")
    print(f"验证集: {len(val_idx):5d} ({len(val_idx) / n * 100:.1f}%)")
    print(f"测试集: {len(test_idx):5d} ({len(test_idx) / n * 100:.1f}%)")

    # ---- 保存索引 ----
    np.save(DATA_DIR / "indices_train.npy", train_idx)
    np.save(DATA_DIR / "indices_val.npy", val_idx)
    np.save(DATA_DIR / "indices_test.npy", test_idx)

    # ---- 保存标签矩阵 ----
    np.save(DATA_DIR / "labels_train.npy", train_labels)
    np.save(DATA_DIR / "labels_val.npy", val_labels)
    np.save(DATA_DIR / "labels_test.npy", test_labels)

    print(f"\n[保存] data/indices_train.npy  shape={train_idx.shape}")
    print(f"[保存] data/indices_val.npy    shape={val_idx.shape}")
    print(f"[保存] data/indices_test.npy   shape={test_idx.shape}")
    print(f"[保存] data/labels_train.npy   shape={train_labels.shape}")
    print(f"[保存] data/labels_val.npy     shape={val_labels.shape}")
    print(f"[保存] data/labels_test.npy    shape={test_labels.shape}")

    # ---- 保存标签元信息 ----
    label_info = {
        "main_labels": MAIN_LABELS,
        "aux_labels": AUX_LABELS,
        "all_labels": ALL_LABELS,
        "multi_label_map": MULTI_LABEL_MAP,
        "label_dim": len(ALL_LABELS),
        "random_state": RANDOM_STATE,
        "split_ratio": {"train": 0.8, "val": 0.1, "test": 0.1},
        "split_counts": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
    }
    with open(DATA_DIR / "label_info.json", "w", encoding="utf-8") as f:
        json.dump(label_info, f, ensure_ascii=False, indent=2)
    print(f"[保存] data/label_info.json")


# ============================================================
# 主流程
# ============================================================
def main():
    print("R 角色 - 阶段一：数据准备")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"随机种子: {RANDOM_STATE}\n")

    ds = download_eurosat()
    train_ds = eda(ds)
    multi_labels = construct_multilabel(train_ds)
    split_dataset(multi_labels)

    print("\n" + "=" * 60)
    print("阶段一完成！交付物：")
    print("=" * 60)
    print("  data/labels_train.npy  (N_train, 17)")
    print("  data/labels_val.npy    (N_val, 17)")
    print("  data/labels_test.npy   (N_test, 17)")
    print("  data/indices_*.npy     划分索引（供 W/L 对齐）")
    print("  data/label_info.json   标签映射信息")
    print("  figures/eda_*.png      EDA 可视化")
    print("\n下一步: 通知 L 和 H 数据集已就绪")


if __name__ == "__main__":
    main()
