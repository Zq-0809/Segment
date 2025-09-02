import cv2
import os
import numpy as np

# 设置原图文件夹和输出文件夹
input_folder = r"D:\datasets\AWSD2025\training\annotations-yuan"             # 原图文件夹路径
output_folder = r"D:\datasets\AWSD2025\training\annotations"    # 输出文件夹路径

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 支持的图像扩展名
supported_exts = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']

# 遍历图像文件
for filename in os.listdir(input_folder):
    name, ext = os.path.splitext(filename)
    if ext.lower() in supported_exts:
        input_path = os.path.join(input_folder, filename)

        # 读取为灰度图
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"跳过无法读取的图像: {filename}")
            continue

        # 使用最近邻插值放大 4 倍
        resized_img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)

        # 保存放大图
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, resized_img)

        # 统计灰度值
        unique_values = np.unique(resized_img)
        print(f"{filename} 的灰度值种类: {len(unique_values)}，灰度值为: {unique_values}")

print("全部图像已处理完毕。")
