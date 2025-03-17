

import os
import shutil

# 指定文件夹A和文件夹B的路径
# folder_A = 'D:/works/studio/data/RegDB/RegDB/split/bounding_box_test/Visible/'
# folder_B = 'D:/works/studio/data/RegDB/RegDB/split/deal/bounding_box_test/Visible/'

# folder_A = 'D:/works/studio/data/RegDB/RegDB/split/combime/bounding_box_test/'
# folder_B = 'D:/works/studio/data/RegDB/RegDB/split/combime/deal/bounding_box_test/'

folder_A = 'D:/works/studio/data/SYSU-MM01/SYSU-MM01/thermal/test/cam3/'
folder_B = 'D:/works/studio/data/SYSU-MM01/SYSU-MM01/deal/thermal/test/cam3/'

# folder_A = 'D:/works/studio/data/SYSU-MM01/SYSU-MM01/visible/test/cam5/'
# folder_B = 'D:/works/studio/data/SYSU-MM01/SYSU-MM01/deal/visible/test/cam5/'

# 确保输出根文件夹存在
if not os.path.exists(folder_B):
    os.makedirs(folder_B)

# 用户指定的cam值
# cam_value = input("请输入cam的值（例如c1_s1）：")
cam_value = "c3_s1"
# 遍历文件夹A中的子文件夹
for folder_K in os.listdir(folder_A):
    # 检查子文件夹是否是数字命名的
    if folder_K.isdigit():
        # 计算文件夹K的pid，并格式化为四位宽度的字符串
        pid = folder_K.zfill(4)

        # 获取文件夹K中的所有图片文件
        # image_files = [f for f in os.listdir(os.path.join(folder_A, folder_K)) if f.endswith('.bmp')]  ##注意图片格式!!!!!!!!!!!!!!!!
        image_files = [f for f in os.listdir(os.path.join(folder_A, folder_K)) if f.endswith('.jpg')]##注意图片格式!!!!!!!!!!!!!!!!

        # 遍历文件夹K中的图片文件
        for i, image_file in enumerate(image_files, start=1):
            # 构建新的文件名
            length = str(i).zfill(6)  # 使用六位宽度表示图片序号
            # new_filename = f"{pid}_{cam_value}_{length}_01.bmp"
            new_filename = f"{pid}_{cam_value}_{length}_01.jpg"

            # 构建源文件和目标文件的完整路径
            source_path = os.path.join(folder_A, folder_K, image_file)
            target_path = os.path.join(folder_B, new_filename)

            # 复制文件并重命名
            shutil.copy2(source_path, target_path)

            # 输出复制文件的信息
            print(f"复制文件: {source_path} 到 {target_path}")

print("任务完成！")
