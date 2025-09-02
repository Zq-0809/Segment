# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 19:48:43 2025

@author: zhouyc
"""
import os

# 设置两个文件夹路径
tiff_folder = r"D:\RSD\AWSD2025\testing\images"  # 包含.tif/.tiff文件
png_folder = r"D:\RSD\AWSD2025\testing\annotations"    # 包含.png文件

# 获取第一个文件夹中所有.tif/.tiff文件的**不带扩展名的文件名集合**
tiff_names = {
    os.path.splitext(f)[0]
    for f in os.listdir(tiff_folder)
    if f.lower().endswith(('.tif', '.tiff'))
}

# 遍历第二个文件夹，删除不在保留名单中的.png文件
for png_file in os.listdir(png_folder):
    if png_file.lower().endswith('.png'):
        png_name = os.path.splitext(png_file)[0]
        if png_name not in tiff_names:
            file_path = os.path.join(png_folder, png_file)
            os.remove(file_path)
            print(f"删除：{file_path}")


