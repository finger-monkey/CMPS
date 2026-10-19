
import os
import shutil

# 定义文件夹A和文件夹B的路径
# folder_A = 'D:/works/studio/data/SYSU-MM01/SYSU-MM01/visible/cam1/'  # 替换为文件夹A的实际路径
# folder_B = 'D:/works/studio/data/SYSU-MM01/SYSU-MM01/visible/test/cam1/'  # 替换为文件夹B的实际路径

folder_A = 'D:/works/studio/data/SYSU-MM01/SYSU-MM01/thermal/cam6/'  # 替换为文件夹A的实际路径
folder_B = 'D:/works/studio/data/SYSU-MM01/SYSU-MM01/thermal/test/cam6/'  # 替换为文件夹B的实际路径

# 确保输出根文件夹存在
if not os.path.exists(folder_B):
    os.makedirs(folder_B)

# 打开并读取test_id.txt文件，获取测试ID列表
with open('D:/works/studio/data/SYSU-MM01/SYSU-MM01/test_id.txt', 'r') as file:
    test_ids = [int(id.strip()) for id in file.readline().split(',')]

# 遍历文件夹A中的子文件夹
for root, dirs, files in os.walk(folder_A):
    for folder_name in dirs:
        # 获取子文件夹的编号
        folder_number = int(folder_name)

        # 如果子文件夹的编号在测试ID列表中
        if folder_number in test_ids:
            # 构建源文件夹和目标文件夹的完整路径
            source_folder_path = os.path.join(root, folder_name)
            target_folder_path = os.path.join(folder_B, folder_name)

            # 移动子文件夹到文件夹B中
            shutil.move(source_folder_path, target_folder_path)
            print(f"移动子文件夹 {folder_name} 到 {target_folder_path}")

print("任务完成！")
