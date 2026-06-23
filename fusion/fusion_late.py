"""
fusion_late.py
R 角色 - 阶段三：晚融合实验

功能：
1. 加载各模型在测试集上的预测概率（由 L 和 W 提供）
   - SVM / RandomForest / XGBoost（L 提供）
   - CNN / ResNet50（W 提供）
2. 基于各模型在结果 CSV 中的 mAP 计算权重
3. 加权投票融合各模型预测概率
4. 在测试集上评估融合结果

前置依赖：
- L 需提供 predictions/predictions_svm_test.npy 等
- W 需提供 predictions/predictions_cnn_test.npy 等
- results/ml_results.csv 和 results/dl_results.csv 已存在

交付物：
- results/fusion_late_results.csv  晚融合实验结果
- results/model_weights.json       各模型权重
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from sklearn.metrics import (
    accuracy_score, hamming_loss, f1_score, average_precision_score
)

# ============================================================
# 配置
# ============================================================
RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_DIR = PROJECT_ROOT / "features"
RESULTS_DIR = PROJECT_ROOT / "results"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
PREDICTIONS_DIR.mkdir(exist_ok=True)

LABEL_DIM = 17


# ============================================================
# 评估函数
# ============================================================
def evaluate_multilabel(y_true, y_pred, y_prob=None):
    """评估多标签分类结果。"""
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
    print("R 角色 - 阶段三：晚融合实验")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"随机种子: {RANDOM_STATE}\n")

    # ---- 步骤 1: 加载标签 ----
    print("=" * 60)
    print("步骤 1: 加载测试集标签")
    print("=" * 60)

    labels_test = np.load(DATA_DIR / "labels_test.npy")
    print(f"  标签 shape: {labels_test.shape}")

    # ---- 步骤 2: 加载各模型预测概率 ----
    print("\n" + "=" * 60)
    print("步骤 2: 加载各模型预测概率")
    print("=" * 60)

    # 定义模型名称与对应文件
    model_files = {
        'SVM': 'predictions_svm_test.npy',
        'RandomForest': 'predictions_rf_test.npy',
        'XGBoost': 'predictions_xgb_test.npy',
        'CNN': 'predictions_cnn_test.npy',
        'ResNet50': 'predictions_resnet_test.npy',
    }

    test_probs = {}
    missing_models = []

    for model_name, file_name in model_files.items():
        file_path = PREDICTIONS_DIR / file_name
        if file_path.exists():
            prob = np.load(file_path)
            test_probs[model_name] = prob
            print(f"  {model_name:15s}: {file_name} ✓  shape={prob.shape}")
        else:
            missing_models.append(model_name)
            print(f"  {model_name:15s}: {file_name} ✗ [缺失]")

    if missing_models:
        print(f"\n  [警告] 缺失 {len(missing_models)} 个模型的预测概率文件:")
        for m in missing_models:
            print(f"    - {m}")
        print(f"\n  请通知 L 和 W 补充交付物（参见 补充交付物说明.md）")

        # 如果有部分模型缺失，仅使用已有模型
        if len(test_probs) == 0:
            print(f"\n  [错误] 没有任何预测概率文件，无法进行晚融合")
            print(f"  请等待 L 和 W 补充交付物后再运行此脚本")
            return

        print(f"\n  将仅使用已有 {len(test_probs)} 个模型进行融合")

    # ---- 步骤 3: 从结果 CSV 读取各模型 mAP 作为权重依据 ----
    print("\n" + "=" * 60)
    print("步骤 3: 基于结果 CSV 中的 mAP 计算权重")
    print("=" * 60)

    # 读取 ml_results.csv
    ml_path = RESULTS_DIR / "ml_results.csv"
    dl_path = RESULTS_DIR / "dl_results.csv"

    model_maps = {}

    if ml_path.exists():
        ml_df = pd.read_csv(ml_path)
        for _, row in ml_df.iterrows():
            model_name = row['model']
            if model_name in test_probs:
                model_maps[model_name] = row['mAP']
                print(f"  {model_name:15s} mAP: {row['mAP']:.6f} (from ml_results.csv)")

    if dl_path.exists():
        dl_df = pd.read_csv(dl_path)
        for _, row in dl_df.iterrows():
            model_name = row['model']
            if model_name in test_probs:
                model_maps[model_name] = row['mAP']
                print(f"  {model_name:15s} mAP: {row['mAP']:.6f} (from dl_results.csv)")

    # 计算归一化权重
    total_map = sum(model_maps.values())
    weights = {name: map_val / total_map for name, map_val in model_maps.items()}

    print(f"\n  归一化权重:")
    for name, w in weights.items():
        print(f"    {name:15s}: {w:.4f}")

    # 保存权重
    weight_info = {
        "weights": weights,
        "model_maps": model_maps,
        "weighting_method": "normalized_test_mAP",
        "used_models": list(test_probs.keys()),
        "missing_models": missing_models,
    }
    with open(RESULTS_DIR / "model_weights.json", "w", encoding="utf-8") as f:
        json.dump(weight_info, f, ensure_ascii=False, indent=2)
    print(f"\n  [保存] results/model_weights.json")

    # ---- 步骤 4: 加权投票融合 ----
    print("\n" + "=" * 60)
    print("步骤 4: 加权投票融合")
    print("=" * 60)

    # 加权平均概率
    fused_prob = np.zeros((len(labels_test), LABEL_DIM), dtype=np.float32)
    for name, prob in test_probs.items():
        fused_prob += weights[name] * prob

    # 阈值 0.5 转为标签
    fused_pred = (fused_prob >= 0.5).astype(np.uint8)

    print(f"  融合概率 shape: {fused_prob.shape}")
    print(f"  使用模型数: {len(test_probs)}")

    # ---- 步骤 5: 评估融合结果 ----
    print("\n" + "=" * 60)
    print("步骤 5: 评估晚融合结果")
    print("=" * 60)

    metrics = evaluate_multilabel(labels_test, fused_pred, fused_prob)

    print(f"\n  晚融合 (加权投票) 测试集指标:")
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
        "model": "LateFusion_Weighted",
        "hamming_loss": metrics["hamming_loss"],
        "macro_f1": metrics["macro_f1"],
        "micro_f1": metrics["micro_f1"],
        "mAP": metrics["mAP"],
        "subset_accuracy": metrics["subset_accuracy"],
    }])
    result_df.to_csv(RESULTS_DIR / "fusion_late_results.csv", index=False)
    print(f"  [保存] results/fusion_late_results.csv")

    print("\n" + "=" * 60)
    print("晚融合实验完成！")
    print("=" * 60)

    if missing_models:
        print(f"\n  注意: 当前仅使用 {len(test_probs)} 个模型，缺失 {len(missing_models)} 个")
        print(f"  补充完整后重新运行可获得更准确的融合结果")


if __name__ == "__main__":
    main()