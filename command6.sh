#!/bin/bash
# R角色 - 阶段三：融合实验
# 此脚本在阿里云DSW上运行

cd /mnt/workspace/terravision

# ============================================================
# 步骤 1: 解压各成员交付的zip文件
# ============================================================
echo "============================================================"
echo "步骤 1: 解压各成员交付的zip文件"
echo "============================================================"

# 解压 W 的 ResNet 特征（用于早融合）
if [ -f "W-stage2-to-R.zip" ]; then
    echo "解压 W-stage2-to-R.zip..."
    unzip -o W-stage2-to-R.zip
else
    echo "[警告] W-stage2-to-R.zip 不存在"
fi

# 解压 W 的补充交付物（预测概率，用于晚融合）
if [ -f "W-stage3-supplement-to-R.zip" ]; then
    echo "解压 W-stage3-supplement-to-R.zip..."
    unzip -o W-stage3-supplement-to-R.zip
else
    echo "[警告] W-stage3-supplement-to-R.zip 不存在"
fi

# 解压 L 的交付物（ML结果和预测概率）
if [ -f "L-stage2-to-R.zip" ]; then
    echo "解压 L-stage2-to-R.zip..."
    unzip -o L-stage2-to-R.zip
else
    echo "[警告] L-stage2-to-R.zip 不存在"
fi

# 解压 L 的阶段2交付物（如果存在）
if [ -f "L-stage2.zip" ]; then
    echo "解压 L-stage2.zip..."
    unzip -o L-stage2.zip
fi

# 解压融合输入文件（如果存在）
if [ -f "fusion-inputs.zip" ]; then
    echo "解压 fusion-inputs.zip..."
    unzip -o fusion-inputs.zip
fi

# ============================================================
# 步骤 2: 检查解压后的文件结构
# ============================================================
echo ""
echo "============================================================"
echo "步骤 2: 检查文件结构"
echo "============================================================"

echo "features/ 目录:"
ls -la features/

echo ""
echo "predictions/ 目录:"
ls -la predictions/

echo ""
echo "results/ 目录:"
ls -la results/

echo ""
echo "data/ 目录:"
ls -la data/

# ============================================================
# 步骤 3: 运行早融合实验
# ============================================================
echo ""
echo "============================================================"
echo "步骤 3: 运行早融合实验"
echo "============================================================"

python fusion/fusion_early.py

# ============================================================
# 步骤 4: 运行晚融合实验
# ============================================================
echo ""
echo "============================================================"
echo "步骤 4: 运行晚融合实验"
echo "============================================================"

python fusion/fusion_late.py

# ============================================================
# 步骤 5: 汇总所有结果
# ============================================================
echo ""
echo "============================================================"
echo "步骤 5: 汇总所有模型结果"
echo "============================================================"

python fusion/generate_all_results.py

# ============================================================
# 步骤 6: 查看最终结果
# ============================================================
echo ""
echo "============================================================"
echo "步骤 6: 查看最终结果"
echo "============================================================"

echo "all_results.csv 内容:"
cat results/all_results.csv

# ============================================================
# 步骤 7: 生成阶段三交付物压缩包
# ============================================================
echo ""
echo "============================================================"
echo "步骤 7: 生成阶段三交付物压缩包"
echo "============================================================"

echo "生成 R-stage3-to-H.zip..."
zip -r R-stage3-to-H.zip \
    results/all_results.csv \
    results/fusion_early_results.csv \
    results/fusion_late_results.csv \
    results/model_weights.json \
    features/early_fusion_train.npy \
    features/early_fusion_test.npy

echo ""
echo "生成 R-stage3.zip（全员分发）..."
zip -r R-stage3.zip \
    results/all_results.csv \
    results/fusion_early_results.csv \
    results/fusion_late_results.csv \
    results/model_weights.json \
    features/early_fusion_train.npy \
    features/early_fusion_test.npy \
    figures/tsne_features.png

# ============================================================
# 步骤 8: 显示交付物信息
# ============================================================
echo ""
echo "============================================================"
echo "步骤 8: 交付物压缩包列表"
echo "============================================================"

ls -lh R-stage3*.zip

echo ""
echo "压缩包内容预览:"
echo ""
echo "--- R-stage3-to-H.zip ---"
unzip -l R-stage3-to-H.zip

echo ""
echo "--- R-stage3.zip ---"
unzip -l R-stage3.zip

echo ""
echo "============================================================"
echo "✓ 阶段三完成！"
echo "============================================================"

# ============================================================
# 步骤 9: 推送结果到远程仓库
# ============================================================
echo ""
echo "============================================================"
echo "步骤 9: 推送结果到远程仓库"
echo "============================================================"

# 添加生成的文件到Git
git add results/*.csv
git add results/*.json
git add features/early_fusion_*.npy
git add R-stage3*.zip
git add command6.log

# 提交
git commit -m "R角色阶段三融合实验完成"

# 推送
git push

echo ""
echo "============================================================"
echo "✓ 推送完成！本地可通过 git pull 获取结果"
echo "============================================================"