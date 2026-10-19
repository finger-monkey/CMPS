

import os
import random
import shutil

# 指定文件夹A和文件夹B的路径
folder_A = 'D:/works/studio/data/RegDB/RegDB/split/bounding_box_test/Visible/'
folder_B = 'D:/works/studio/data/RegDB/RegDB/split/bounding_box_test/Visible2/'

# 获取文件夹A中的所有子文件夹
subfolders_A = [f for f in os.listdir(folder_A) if os.path.isdir(os.path.join(folder_A, f))]

# 遍历每个子文件夹
for folder_i in subfolders_A:
    # 构建文件夹i的完整路径
    folder_A_i = os.path.join(folder_A, folder_i)

    # 获取文件夹i中的所有图片文件
    image_files = [f for f in os.listdir(folder_A_i) if f.endswith('.bmp')] #注意图片格式

    # 计算要剪切的图片数量（一半）
    num_images_to_cut = len(image_files) // 2

    # 随机选择要剪切的图片
    images_to_cut = random.sample(image_files, num_images_to_cut)

    # 构建文件夹i在文件夹B中的路径
    folder_B_i = os.path.join(folder_B, folder_i)

    # 确保文件夹B中的文件夹i存在
    if not os.path.exists(folder_B_i):
        os.makedirs(folder_B_i)

    # 遍历要剪切的图片并执行剪切操作
    for image in images_to_cut:
        source_path = os.path.join(folder_A_i, image)
        target_path = os.path.join(folder_B_i, image)

        # 执行剪切操作
        shutil.move(source_path, target_path)

        # 输出剪切文件的信息
        print(f"剪切文件: {source_path} 到 {target_path}")

print("任务完成！")
