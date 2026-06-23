"""
generate_all_results.py
R 角色 - 阶段三：汇总所有模型结果

功能：
1. 读取 ml_results.csv（L 交付）
2. 读取 dl_results.csv（W 交付）
3. 读取 fusion_early_results.csv（R 早融合）
4. 读取 fusion_late_results.csv（R 晚融合）
5. 汇总为 all_results.csv，统一格式

交付物：
- results/all_results.csv  汇总所有模型（含融合）的完整指标
"""

import pandas as pd
from pathlib import Path

# ============================================================
# 配置
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


# ============================================================
# 主流程
# ============================================================
def main():
    print("R 角色 - 阶段三：汇总所有模型结果")
    print(f"项目根目录: {PROJECT_ROOT}\n")

    print("=" * 60)
    print("步骤 1: 读取各结果文件")
    print("=" * 60)

    all_results = []

    # 1. 传统 ML 结果（L 交付）
    ml_path = RESULTS_DIR / "ml_results.csv"
    if ml_path.exists():
        ml_df = pd.read_csv(ml_path)
        print(f"  读取 ml_results.csv: {len(ml_df)} 条记录")
        # 确保列一致
        if 'subset_accuracy' not in ml_df.columns:
            ml_df['subset_accuracy'] = None
        all_results.append(ml_df)
    else:
        print(f"  [警告] 未找到 ml_results.csv")

    # 2. 深度学习结果（W 交付）
    dl_path = RESULTS_DIR / "dl_results.csv"
    if dl_path.exists():
        dl_df = pd.read_csv(dl_path)
        print(f"  读取 dl_results.csv: {len(dl_df)} 条记录")
        if 'subset_accuracy' not in dl_df.columns:
            dl_df['subset_accuracy'] = None
        all_results.append(dl_df)
    else:
        print(f"  [警告] 未找到 dl_results.csv")

    # 3. 早融合结果（R 交付）
    early_path = RESULTS_DIR / "fusion_early_results.csv"
    if early_path.exists():
        early_df = pd.read_csv(early_path)
        print(f"  读取 fusion_early_results.csv: {len(early_df)} 条记录")
        all_results.append(early_df)
    else:
        print(f"  [警告] 未找到 fusion_early_results.csv")

    # 4. 晚融合结果（R 交付）
    late_path = RESULTS_DIR / "fusion_late_results.csv"
    if late_path.exists():
        late_df = pd.read_csv(late_path)
        print(f"  读取 fusion_late_results.csv: {len(late_df)} 条记录")
        all_results.append(late_df)
    else:
        print(f"  [警告] 未找到 fusion_late_results.csv")

    # ---- 合并 ----
    print("\n" + "=" * 60)
    print("步骤 2: 合并结果")
    print("=" * 60)

    if not all_results:
        print("  [错误] 没有找到任何结果文件")
        return

    # 统一列顺序
    column_order = ['model', 'hamming_loss', 'macro_f1', 'micro_f1', 'mAP', 'subset_accuracy']
    all_df = pd.concat(all_results, ignore_index=True)

    # 确保所有列都存在
    for col in column_order:
        if col not in all_df.columns:
            all_df[col] = None

    all_df = all_df[column_order]

    # 按 mAP 降序排序
    all_df = all_df.sort_values('mAP', ascending=False).reset_index(drop=True)

    print(f"\n  合并后共 {len(all_df)} 个模型结果:")
    print(all_df.to_string(index=False))

    # ---- 保存 ----
    print("\n" + "=" * 60)
    print("步骤 3: 保存汇总结果")
    print("=" * 60)

    output_path = RESULTS_DIR / "all_results.csv"
    all_df.to_csv(output_path, index=False)
    print(f"\n  [保存] results/all_results.csv")
    print(f"  共 {len(all_df)} 个模型，按 mAP 降序排列")

    # ---- 打印汇总信息 ----
    print("\n" + "=" * 60)
    print("汇总完成！模型性能排名:")
    print("=" * 60)
    for i, row in all_df.iterrows():
        print(f"  {i+1}. {row['model']:25s}  mAP={row['mAP']:.4f}  Micro F1={row['micro_f1']:.4f}")

    print("\n下一步: 将 all_results.csv 交给 H 进行可视化")


if __name__ == "__main__":
    main()
