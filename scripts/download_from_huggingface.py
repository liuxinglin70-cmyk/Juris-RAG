"""
从 Hugging Face Hub 下载数据集
"""
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download

def download_dataset(repo_id: str, local_dir: str = "data"):
    """
    从Hugging Face下载完整数据集
    
    Args:
        repo_id: Hugging Face数据集ID，格式为 "username/dataset-name"
        local_dir: 本地保存目录
    """
    print(f"📥 从 {repo_id} 下载数据集...")
    
    # 方法1：下载整个数据集
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        ignore_patterns=["*.md"]  # 忽略README
    )
    
    print(f"✅ 数据集已下载到: {Path(local_dir).absolute()}")

def download_specific_file(repo_id: str, filename: str, local_dir: str = "data"):
    """
    下载特定文件
    
    Args:
        repo_id: Hugging Face数据集ID
        filename: 文件路径，如 "raw/cail_cases.json"
        local_dir: 本地保存目录
    """
    print(f"📥 下载文件: {filename}")
    
    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=local_dir
    )
    
    print(f"✅ 文件已下载到: {file_path}")
    return file_path

def main():
    # 配置
    repo_id = input("输入Hugging Face数据集ID (格式: username/dataset-name): ")
    
    choice = input("\n选择下载方式:\n1. 下载完整数据集\n2. 下载特定文件\n输入选择 (1/2): ")
    
    if choice == "1":
        download_dataset(repo_id)
    elif choice == "2":
        filename = input("输入文件路径 (如 raw/cail_cases.json): ")
        download_specific_file(repo_id, filename)
    else:
        print("无效选择")

if __name__ == "__main__":
    main()
