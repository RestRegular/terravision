"""
feature_reduce.py
R 角色 - 阶段二：特征降维与可视化

功能：
1. 加载 features_raw_*.npy（未降维的手工特征）
2. PCA 降维至 128 维（使用训练集拟合，转换验证集和测试集）
3. 保存降维后的特征矩阵（供 L 使用）
4. t-SNE 可视化特征分布（在测试集上，按主标签着色）
5. 保存 PCA 模型（供融合实验复用）

交付物：
- features/features_train.npy   (N_train, 128)
- features/features_val.npy     (N_val, 128)
- features/features_test.npy    (N_test, 128)
- figures/tsne_features.png     t-SNE 特征分布图
- features/pca_model.pkl        PCA 模型（供融合实验使用）
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle
import json

# ============================================================
# 配置
# ============================================================
RANDOM_STATE = 42
PCA_DIM = 128
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_DIR = PROJECT_ROOT / "features"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# 主标签名称（用于 t-SNE 可视化）
MAIN_LABELS = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]


# ============================================================
# 主流程
# ============================================================
def main():
    print("R 角色 - 阶段二：特征降维与可视化")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"随机种子: {RANDOM_STATE}")
    print(f"PCA 目标维度: {PCA_DIM}\n")

    # ---- 步骤 1: 加载原始特征 ----
    print("=" * 60)
    print("步骤 1: 加载原始特征矩阵")
    print("=" * 60)

    train_raw = np.load(FEATURES_DIR / "features_raw_train.npy")
    val_raw = np.load(FEATURES_DIR / "features_raw_val.npy")
    test_raw = np.load(FEATURES_DIR / "features_raw_test.npy")

    print(f"  features_raw_train.npy: {train_raw.shape}")
    print(f"  features_raw_val.npy:   {val_raw.shape}")
    print(f"  features_raw_test.npy:  {test_raw.shape}")

    # ---- 步骤 2: PCA 降维 ----
    print("\n" + "=" * 60)
    print("步骤 2: PCA 降维")
    print("=" * 60)

    # 使用训练集拟合 PCA
    pca = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE)
    train_pca = pca.fit_transform(train_raw)

    # 转换验证集和测试集
    val_pca = pca.transform(val_raw)
    test_pca = pca.transform(test_raw)

    print(f"  PCA 拟合完成，保留 {PCA_DIM} 维")
    print(f"  累计方差解释比例: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"  降维后 train shape: {train_pca.shape}")
    print(f"  降维后 val shape:   {val_pca.shape}")
    print(f"  降维后 test shape:  {test_pca.shape}")

    # ---- 步骤 3: 保存降维后的特征 ----
    print("\n" + "=" * 60)
    print("步骤 3: 保存降维后的特征矩阵")
    print("=" * 60)

    np.save(FEATURES_DIR / "features_train.npy", train_pca)
    np.save(FEATURES_DIR / "features_val.npy", val_pca)
    np.save(FEATURES_DIR / "features_test.npy", test_pca)

    print(f"\n[保存] features/features_train.npy  shape={train_pca.shape}")
    print(f"[保存] features/features_val.npy    shape={val_pca.shape}")
    print(f"[保存] features/features_test.npy   shape={test_pca.shape}")

    # ---- 步骤 4: 保存 PCA 模型 ----
    print("\n" + "=" * 60)
    print("步骤 4: 保存 PCA 模型")
    print("=" * 60)

    with open(FEATURES_DIR / "pca_model.pkl", "wb") as f:
        pickle.dump(pca, f)
    print(f"[保存] features/pca_model.pkl")

    # ---- 步骤 5: t-SNE 可视化 ----
    print("\n" + "=" * 60)
    print("步骤 5: t-SNE 可视化（测试集）")
    print("=" * 60)

    # 加载测试集标签（用于着色）
    test_labels = np.load(DATA_DIR / "labels_test.npy")
    # 提取主标签（前 10 维）
    test_main_labels = test_labels[:, :10].argmax(axis=1)

    print(f"  t-SNE 输入: {test_pca.shape}")
    print(f"  开始 t-SNE 计算（可能需要几分钟）...")

    # t-SNE 参数（兼容不同 sklearn 版本）
    try:
        # 新版本 sklearn 使用 max_iter
        tsne = TSNE(
            n_components=2,
            perplexity=30,
            learning_rate=200,
            max_iter=1000,
            random_state=RANDOM_STATE,
            verbose=1,
        )
    except TypeError:
        # 旧版本 sklearn 使用 n_iter
        tsne = TSNE(
            n_components=2,
            perplexity=30,
            learning_rate=200,
            n_iter=1000,
            random_state=RANDOM_STATE,
            verbose=1,
        )
    test_tsne = tsne.fit_transform(test_pca)

    print(f"  t-SNE 输出: {test_tsne.shape}")

    # ---- 绘制 t-SNE 图 ----
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        test_tsne[:, 0],
        test_tsne[:, 1],
        c=test_main_labels,
        cmap="tab10",
        alpha=0.6,
        s=20,
    )
    plt.colorbar(scatter, ticks=range(10), label="Class")
    plt.title("t-SNE Visualization of Handcrafted Features (Test Set)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")

    # 添加类别标签
    cbar = plt.gcf().axes[-1]
    cbar.set_yticklabels(MAIN_LABELS)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tsne_features.png", dpi=150)
    plt.close()

    print(f"\n[保存] figures/tsne_features.png")

    # ---- 步骤 6: 保存降维元信息 ----
    print("\n" + "=" * 60)
    print("步骤 6: 保存降维元信息")
    print("=" * 60)

    reduce_info = {
        "pca_dim": PCA_DIM,
        "explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
        "original_dim": int(train_raw.shape[1]),
        "random_state": RANDOM_STATE,
        "tsne_perplexity": 30,
        "tsne_learning_rate": 200,
        "tsne_n_iter": 1000,
    }
    with open(FEATURES_DIR / "reduce_info.json", "w", encoding="utf-8") as f:
        json.dump(reduce_info, f, ensure_ascii=False, indent=2)
    print(f"[保存] features/reduce_info.json")

    print("\n" + "=" * 60)
    print("阶段二（特征降维）完成！")
    print("=" * 60)
    print("  features/features_train.npy   (N_train, 128)")
    print("  features/features_val.npy     (N_val, 128)")
    print("  features/features_test.npy    (N_test, 128)")
    print("  figures/tsne_features.png     t-SNE 可视化")
    print("  features/pca_model.pkl        PCA 模型")
    print("\n下一步: 通知 L 特征矩阵已就绪，可开始训练传统 ML 模型")


if __name__ == "__main__":
    main()