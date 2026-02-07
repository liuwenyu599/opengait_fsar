import pickle
import sys
import numpy as np
import os
import torch


def inspect_pkl(file_path):
    print(f"\n{'=' * 20} Inspecting: {os.path.basename(file_path)} {'=' * 20}")

    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load pickle: {e}")
        return

    # 1. 检查整体结构
    print(f"[INFO] Data Type: {type(data)}")
    if not isinstance(data, (list, tuple)):
        print("[ERROR] Data is not a list or tuple! OpenGait requires a list of modalities.")
        return

    print(f"[INFO] List Length: {len(data)}")

    # 2. 逐个检查模态
    for i, item in enumerate(data):
        print(f"\n--- Modality {i} ---")
        print(f"Type: {type(item)}")

        # 如果是 Numpy 或 Tensor，打印详细形状
        if isinstance(item, (np.ndarray, torch.Tensor)):
            shape = item.shape
            print(f"Shape: {shape}")
            print(f"Dtype: {item.dtype}")

            # 统计数值范围
            if isinstance(item, torch.Tensor):
                print(f"Min/Max: {item.min():.4f} / {item.max():.4f}")
            else:
                print(f"Min/Max: {item.min():.4f} / {item.max():.4f}")

            # 3. 智能推断：这是轮廓还是骨骼？
            # 逻辑：骨骼通常最后一维是 2, 3, 34, 51 等；轮廓通常是 H*W (44, 64)
            dims = len(shape)
            last_dim = shape[-1]

            if dims >= 3 and last_dim > 10 and last_dim < 60:
                print(">>> [推测] 这可能是【骨骼 (Skeleton)】数据")

                # 关键点数量检查
                if last_dim == 51:
                    print(f"    [OK] 格式符合 17关键点 * 3维 (51)")
                elif last_dim == 34:
                    print(f"    [WARN] 格式是 17关键点 * 2维 (34)。模型可能需要 3维输入！")
                elif last_dim == 17:
                    if dims == 4 and shape[-2] == 3:  # [T, 3, 17]
                        print(f"    [WARN] 维度排列可能是 [T, 3, 17]。模型通常期待 [T, 17, 3] -> reshape(T, 51)")
                    else:
                        print(f"    [WARN] 最后一维是 17。是 17 个点吗？那坐标去哪了？")
                else:
                    print(f"    [WARN] 最后一维是 {last_dim}，既不是 51 也不是 34。请确认关键点数量！")

            elif dims >= 3 and last_dim >= 44:
                print(">>> [推测] 这可能是【轮廓 (Silhouette)】数据")
                if dims == 3:
                    print("    [WARN] 维度是 3维 [T, H, W]。缺少 Channel 维度！")
                    print("           -> 这会导致模型里 tensor.dim()==4 (N,T,H,W)，被 if dim()==5 过滤掉。")
                    print("           -> 建议在 dataset 或 transform 里 unsqueeze(1)。")
                elif dims == 4 and shape[1] == 1:
                    print("    [OK] 维度是 4维 [C, T, H, W] 或 [T, C, H, W]。")

        else:
            print(f"Value: {item}")


if __name__ == "__main__":
    # 使用方法：直接在下面填入你一个真实的 pkl 文件路径
    # 例如：/data/CASIA-B-Fusion/001/nm-01/090/001-nm-01-090.pkl

    target_file = "/home/lwy/projects/OpenGait-master/data/001/bg-01/000/skels-000.pkl"  # <--- 请在这里修改路径

    if target_file is None:
        print("请编辑脚本 inspect_data.py，在 target_file 变量中填入真实文件路径！")
    else:
        inspect_pkl(target_file)