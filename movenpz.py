
import os
import shutil

def move_n_npz_files(source_dir, target_dir, n=10):
    """
    使用os.scandir扫描源文件夹，移动前N个.npz文件到目标文件夹。
    
    参数:
    - source_dir: 源文件夹路径 (str)
    - target_dir: 目标文件夹路径 (str)
    - n: 要移动的文件数量 (int, 默认10)
    
    返回:
    - moved_files: 移动的文件列表
    """
    # 确保目标文件夹存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 扫描源文件夹中的.npz文件
    npz_files = []
    with os.scandir(source_dir) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith('.npz'):
                npz_files.append(entry.path)
    
    # 只移动前N个（如果文件少于N，只移动所有）
    to_move = npz_files[:n]
    moved_files = []
    
    for file_path in to_move:
        try:
            # 构建目标路径（保持原文件名）
            file_name = os.path.basename(file_path)
            target_path = os.path.join(target_dir, file_name)
            
            # 移动文件
            shutil.move(file_path, target_path)
            moved_files.append(file_name)
            print(f"移动成功: {file_name}")
        except Exception as e:
            print(f"移动失败 {file_path}: {e}")
    
    print(f"\n总共移动了 {len(moved_files)} 个文件。")
    return moved_files

# 示例使用
if __name__ == "__main__":
    source_folder = r"D:\WKS\traindata"  # 替换为你的源文件夹路径
    target_folder = r"D:\WKS\Samples"  # 替换为你的目标文件夹路径
    num_files = 50000  # 指定移动的数量
    
    moved = move_n_npz_files(source_folder, target_folder, num_files)
    