"""
fusion_early.py
R 角色 - 阶段三：早融合实验

功能：
1. 加载手工特征（PCA降维后 128 维）和 ResNet 深度特征（2048 维）
2. 特征拼接前对深度特征进行 L2 归一化（手工特征已归一化）
3. 早融合：拼接形成 128 + 2048 = 2176 维融合特征
4. 可选：对融合特征进行 PCA 降维（避免维度过高）
5. 训练分类器（XGBoost，因其在阶段二表现最佳）
6. 在测试集上评估，输出指标

交付物：
- results/fusion_early_results.csv  早融合实验结果
- features/early_fusion_train.npy   (可选) 融合特征矩阵
- features/early_fusion_test.npy    (可选) 融合特征矩阵
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import normalize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    accuracy_score, hamming_loss, f1_score, average_precision_score
)
from xgboost import XGBClassifier
import time

# ============================================================
# 配置
# ============================================================
RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_DIR = PROJECT_ROOT / "features"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# 评估函数
# ============================================================
def evaluate_multilabel(y_true, y_pred, y_prob=None):
    """评估多标签分类结果。

    参数：
        y_true: 真实标签 (N, 17)
        y_pred: 预测标签 (N, 17)
        y_prob: 预测概率 (N, 17)，用于 mAP 计算

    返回：
        指标字典
    """
    metrics = {
        "hamming_loss": hamming_loss(y_true, y_pred),
        "subset_accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average='micro', zero_division=0),
    }
    if y_prob is not None:
        metrics["mAP"] = average_precision_score(y_true, y_prob, average='macro')
    else:
        metrics["mAP"] = 0.0
    return metrics


# ============================================================
# 主流程
# ============================================================
def main():
    print("R 角色 - 阶段三：早融合实验")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"随机种子: {RANDOM_STATE}\n")

    # ---- 步骤 1: 加载特征 ----
    print("=" * 60)
    print("步骤 1: 加载特征矩阵")
    print("=" * 60)

    # 手工特征（PCA降维后，已 L2 归一化）
    handcraft_train = np.load(FEATURES_DIR / "features_train.npy")
    handcraft_test = np.load(FEATURES_DIR / "features_test.npy")

    # ResNet 深度特征
    resnet_train = np.load(FEATURES_DIR / "resnet_features_train.npy")
    resnet_test = np.load(FEATURES_DIR / "resnet_features_test.npy")

    # 标签
    labels_train = np.load(DATA_DIR / "labels_train.npy")
    labels_test = np.load(DATA_DIR / "labels_test.npy")

    print(f"  手工特征 train: {handcraft_train.shape}")
    print(f"  手工特征 test:  {handcraft_test.shape}")
    print(f"  ResNet特征 train: {resnet_train.shape}")
    print(f"  ResNet特征 test:  {resnet_test.shape}")
    print(f"  标签 train: {labels_train.shape}")
    print(f"  标签 test:  {labels_test.shape}")

    # ---- 步骤 2: 深度特征 L2 归一化 ----
    print("\n" + "=" * 60)
    print("步骤 2: 深度特征 L2 归一化")
    print("=" * 60)

    resnet_train_norm = normalize(resnet_train, norm='l2', axis=1)
    resnet_test_norm = normalize(resnet_test, norm='l2', axis=1)

    print(f"  归一化后 ResNet train 范数: {np.linalg.norm(resnet_train_norm[0]):.4f}")

    # ---- 步骤 3: 特征拼接（早融合）----
    print("\n" + "=" * 60)
    print("步骤 3: 特征拼接（早融合）")
    print("=" * 60)

    fusion_train = np.concatenate([handcraft_train, resnet_train_norm], axis=1)
    fusion_test = np.concatenate([handcraft_test, resnet_test_norm], axis=1)

    print(f"  融合特征 train: {fusion_train.shape}")
    print(f"  融合特征 test:  {fusion_test.shape}")

    # 保存融合特征
    np.save(FEATURES_DIR / "early_fusion_train.npy", fusion_train)
    np.save(FEATURES_DIR / "early_fusion_test.npy", fusion_test)
    print(f"  [保存] features/early_fusion_train.npy")
    print(f"  [保存] features/early_fusion_test.npy")

    # ---- 步骤 4: 训练 XGBoost 分类器 ----
    print("\n" + "=" * 60)
    print("步骤 4: 训练 XGBoost 分类器（基于融合特征）")
    print("=" * 60)

    # 使用 XGBoost（阶段二最佳模型）
    base_clf = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric='logloss',
    )
    clf = MultiOutputClassifier(base_clf, n_jobs=-1)

    print(f"  开始训练...")
    start_time = time.time()
    clf.fit(fusion_train, labels_train)
    train_time = time.time() - start_time
    print(f"  训练完成，耗时: {train_time:.1f}s")

    # ---- 步骤 5: 测试集评估 ----
    print("\n" + "=" * 60)
    print("步骤 5: 测试集评估")
    print("=" * 60)

    y_pred = clf.predict(fusion_test)
    # 获取概率预测（用于 mAP）
    y_prob = np.zeros_like(labels_test, dtype=np.float32)
    for i, estimator in enumerate(clf.estimators_):
        y_prob[:, i] = estimator.predict_proba(fusion_test)[:, 1]

    metrics = evaluate_multilabel(labels_test, y_pred, y_prob)

    print(f"\n  早融合 (XGBoost) 测试集指标:")
    print(f"    Hamming Loss:     {metrics['hamming_loss']:.6f}")
    print(f"    Subset Accuracy:  {metrics['subset_accuracy']:.6f}")
    print(f"    Macro F1:         {metrics['macro_f1']:.6f}")
    print(f"    Micro F1:         {metrics['micro_f1']:.6f}")
    print(f"    mAP:              {metrics['mAP']:.6f}")

    # ---- 步骤 6: 保存结果 ----
    print("\n" + "=" * 60)
    print("步骤 6: 保存结果")
    print("=" * 60)

    result_df = pd.DataFrame([{
        "model": "EarlyFusion_XGBoost",
        "hamming_loss": metrics["hamming_loss"],
        "macro_f1": metrics["macro_f1"],
        "micro_f1": metrics["micro_f1"],
        "mAP": metrics["mAP"],
        "subset_accuracy": metrics["subset_accuracy"],
    }])
    result_df.to_csv(RESULTS_DIR / "fusion_early_results.csv", index=False)
    print(f"  [保存] results/fusion_early_results.csv")

    print("\n" + "=" * 60)
    print("早融合实验完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
