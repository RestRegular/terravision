"""
feature_extract.py
R 角色 - 阶段二：手工特征提取

功能：
1. 从 HuggingFace 加载 EuroSAT 原始图像（合并所有 split）
2. 使用 indices_*.npy 对齐训练/验证/测试集划分
3. 提取四种手工特征：
   - 颜色统计特征：RGB各通道均值/标准差/直方图（~96维）
   - 纹理特征：LBP（局部二值模式，~256维）
   - 梯度特征：HOG（方向梯度直方图，~1764维）
   - 频域特征：灰度图FFT幅度统计（~32维）
4. 拼接为统一特征矩阵并进行 L2 归一化
5. 保存为 features_raw_*.npy（未降维，供 feature_reduce.py 使用）

交付物：
- features/features_raw_train.npy  (N_train, ~2148)
- features/features_raw_val.npy    (N_val, ~2148)
- features/features_raw_test.npy   (N_test, ~2148)
"""

import numpy as np
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray
import cv2
from tqdm import tqdm

# ============================================================
# 配置
# ============================================================
RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_DIR = PROJECT_ROOT / "features"
FEATURES_DIR.mkdir(exist_ok=True)

# EuroSAT 在 HuggingFace 上的标识
HF_DATASET_NAME = "timm/eurosat-rgb"

# ============================================================
# 特征提取函数
# ============================================================
def extract_color_features(img: np.ndarray) -> np.ndarray:
    """提取颜色统计特征：RGB各通道均值/标准差/直方图。

    参数：
        img: RGB图像，shape (H, W, 3)，dtype uint8

    返回：
        特征向量，维度约 96（3通道 × (均值+标准差+32bin直方图)）
    """
    features = []
    for c in range(3):  # R, G, B
        channel = img[:, :, c].astype(np.float32)
        # 均值和标准差
        features.append(channel.mean())
        features.append(channel.std())
        # 直方图（32 bins，归一化）
        hist = np.histogram(channel, bins=32, range=(0, 255))[0]
        hist = hist / hist.sum() if hist.sum() > 0 else hist
        features.extend(hist)
    return np.array(features, dtype=np.float32)


def extract_lbp_features(img: np.ndarray) -> np.ndarray:
    """提取 LBP 纹理特征。

    参数：
        img: RGB图像，shape (H, W, 3)，dtype uint8

    返回：
        特征向量，维度 256（LBP 直方图）
    """
    gray = rgb2gray(img)  # 转为灰度，范围 [0, 1]
    gray_uint8 = (gray * 255).astype(np.uint8)

    # LBP 参数：P=8, R=1（标准设置）
    lbp = local_binary_pattern(gray_uint8, P=8, R=1, method='uniform')

    # LBP 直方图（256 bins，因为 method='nri_uniform' 会产生 0-255）
    # 使用 'uniform' 方法，实际只有 59 个唯一值，但我们用 256 bins 以简化
    hist = np.histogram(lbp.ravel(), bins=256, range=(0, 256))[0]
    hist = hist / hist.sum() if hist.sum() > 0 else hist
    return hist.astype(np.float32)


def extract_hog_features(img: np.ndarray) -> np.ndarray:
    """提取 HOG 梯度特征。

    参数：
        img: RGB图像，shape (H, W, 3)，dtype uint8

    返回：
        特征向量，维度约 1764
    """
    gray = rgb2gray(img)

    # HOG 参数（针对 64x64 图像优化）
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        transform_sqrt=True,
        feature_vector=True,
    )
    return hog_features.astype(np.float32)


def extract_fft_features(img: np.ndarray) -> np.ndarray:
    """提取 FFT 频域特征。

    参数：
        img: RGB图像，shape (H, W, 3)，dtype uint8

    返回：
        特征向量，维度约 32（FFT 幅度统计）
    """
    gray = rgb2gray(img)

    # FFT 变换
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)

    # 提取统计特征
    features = []
    # 全局统计
    features.append(magnitude.mean())
    features.append(magnitude.std())
    features.append(magnitude.max())
    features.append(magnitude.min())

    # 中心区域（低频）统计
    h, w = magnitude.shape
    center_h, center_w = h // 4, w // 4
    center_region = magnitude[
        h//2 - center_h//2 : h//2 + center_h//2,
        w//2 - center_w//2 : w//2 + center_w//2
    ]
    features.append(center_region.mean())
    features.append(center_region.std())

    # 高频区域统计（边缘）
    high_freq = magnitude[
        :h//8, :w//8
    ]
    features.append(high_freq.mean())
    features.append(high_freq.std())

    # 对数幅度直方图（24 bins）
    log_magnitude = np.log(magnitude + 1e-8)
    hist = np.histogram(log_magnitude.ravel(), bins=24, range=(0, log_magnitude.max()))[0]
    hist = hist / hist.sum() if hist.sum() > 0 else hist
    features.extend(hist)

    return np.array(features, dtype=np.float32)


def extract_all_features(img: np.ndarray) -> np.ndarray:
    """提取所有手工特征并拼接。

    参数：
        img: RGB图像，shape (H, W, 3)，dtype uint8

    返回：
        拼接后的特征向量，维度约 2148
    """
    color_feat = extract_color_features(img)
    lbp_feat = extract_lbp_features(img)
    hog_feat = extract_hog_features(img)
    fft_feat = extract_fft_features(img)

    # 拼接
    all_feat = np.concatenate([color_feat, lbp_feat, hog_feat, fft_feat])
    return all_feat


def l2_normalize(features: np.ndarray) -> np.ndarray:
    """L2 归一化特征矩阵。

    参数：
        features: shape (N, D)

    返回：
        归一化后的特征，shape (N, D)
    """
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1  # 防止除零
    return features / norms


# ============================================================
# 主流程
# ============================================================
def main():
    print("R 角色 - 阶段二：手工特征提取")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"随机种子: {RANDOM_STATE}\n")

    # ---- 步骤 1: 加载原始图像 ----
    print("=" * 60)
    print("步骤 1: 加载 EuroSAT 原始图像")
    print("=" * 60)

    print(f"  加载: {HF_DATASET_NAME}")
    ds = load_dataset(HF_DATASET_NAME)
    merged = concatenate_datasets(list(ds.values()))
    print(f"  合并后总样本数: {len(merged)}")

    # ---- 步骤 2: 加载划分索引 ----
    print("\n" + "=" * 60)
    print("步骤 2: 加载划分索引")
    print("=" * 60)

    train_idx = np.load(DATA_DIR / "indices_train.npy")
    val_idx = np.load(DATA_DIR / "indices_val.npy")
    test_idx = np.load(DATA_DIR / "indices_test.npy")

    print(f"  训练集索引: {len(train_idx)}")
    print(f"  验证集索引: {len(val_idx)}")
    print(f"  测试集索引: {len(test_idx)}")

    # ---- 步骤 3: 提取特征 ----
    print("\n" + "=" * 60)
    print("步骤 3: 提取手工特征")
    print("=" * 60)

    def extract_features_for_indices(indices, split_name):
        """为指定索引提取特征。"""
        print(f"\n  提取 {split_name} 特征 ({len(indices)} 张图像)...")
        features_list = []
        for i in tqdm(indices, desc=f"  {split_name}"):
            img = np.array(merged[i]["image"])
            feat = extract_all_features(img)
            features_list.append(feat)
        features = np.array(features_list, dtype=np.float32)
        print(f"    特征矩阵 shape: {features.shape}")
        return features

    train_features = extract_features_for_indices(train_idx, "train")
    val_features = extract_features_for_indices(val_idx, "val")
    test_features = extract_features_for_indices(test_idx, "test")

    # ---- 步骤 4: L2 归一化 ----
    print("\n" + "=" * 60)
    print("步骤 4: L2 归一化")
    print("=" * 60)

    train_features = l2_normalize(train_features)
    val_features = l2_normalize(val_features)
    test_features = l2_normalize(test_features)

    print(f"  归一化后 train shape: {train_features.shape}")
    print(f"  归一化后 val shape: {val_features.shape}")
    print(f"  归一化后 test shape: {test_features.shape}")

    # ---- 步骤 5: 保存特征矩阵 ----
    print("\n" + "=" * 60)
    print("步骤 5: 保存特征矩阵")
    print("=" * 60)

    np.save(FEATURES_DIR / "features_raw_train.npy", train_features)
    np.save(FEATURES_DIR / "features_raw_val.npy", val_features)
    np.save(FEATURES_DIR / "features_raw_test.npy", test_features)

    print(f"\n[保存] features/features_raw_train.npy  shape={train_features.shape}")
    print(f"[保存] features/features_raw_val.npy    shape={val_features.shape}")
    print(f"[保存] features/features_raw_test.npy   shape={test_features.shape}")

    # ---- 特征维度统计 ----
    print("\n" + "=" * 60)
    print("特征维度统计")
    print("=" * 60)

    # 提取一张样本图像，统计各特征维度
    sample_img = np.array(merged[0]["image"])
    color_dim = len(extract_color_features(sample_img))
    lbp_dim = len(extract_lbp_features(sample_img))
    hog_dim = len(extract_hog_features(sample_img))
    fft_dim = len(extract_fft_features(sample_img))

    print(f"  颜色统计特征: {color_dim} 维")
    print(f"  LBP 纹理特征: {lbp_dim} 维")
    print(f"  HOG 梯度特征: {hog_dim} 维")
    print(f"  FFT 频域特征: {fft_dim} 维")
    print(f"  总计: {color_dim + lbp_dim + hog_dim + fft_dim} 维")

    print("\n" + "=" * 60)
    print("阶段二（特征提取）完成！")
    print("=" * 60)
    print("  features/features_raw_train.npy  (N_train, ~2148)")
    print("  features/features_raw_val.npy    (N_val, ~2148)")
    print("  features/features_raw_test.npy   (N_test, ~2148)")
    print("\n下一步: 运行 feature_reduce.py 进行 PCA 降维和 t-SNE 可视化")


if __name__ == "__main__":
    main()