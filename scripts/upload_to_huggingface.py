"""
上传数据集到 Hugging Face Hub
"""
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, login

def upload_dataset():
    """上传Juris-RAG数据集到Hugging Face"""
    
    # 1. 登录（首次需要提供token）
    import os
    
    # 检查是否已经登录或有token
    token = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    
    if not token:
        print("\n🔑 需要Hugging Face访问令牌")
        print("获取方式: https://huggingface.co/settings/tokens")
        print("\n选择登录方式:")
        print("1. 输入访问令牌（推荐）")
        print("2. 通过浏览器登录")
        
        choice = input("\n选择 (1/2): ").strip()
        
        if choice == "1":
            token = input("\n请粘贴你的访问令牌: ").strip()
            if token:
                try:
                    login(token=token, add_to_git_credential=True)
                    print("✅ 登录成功！")
                except Exception as e:
                    print(f"❌ 登录失败: {e}")
                    return
            else:
                print("❌ 未提供令牌")
                return
        elif choice == "2":
            try:
                # 使用notebook=False来避免在非notebook环境中的问题
                login(add_to_git_credential=True)
                print("✅ 登录成功！")
            except Exception as e:
                print(f"❌ 登录失败: {e}")
                print("\n提示: 如果浏览器登录失败，请选择方式1手动输入令牌")
                return
        else:
            print("无效选择")
            return
    else:
        print("✅ 检测到已有访问令牌")
    
    # 2. 配置
    username = input("输入你的Hugging Face用户名: ")
    repo_id = f"{username}/juris-rag-dataset"
    
    api = HfApi()
    
    # 3. 创建数据集仓库
    print(f"\n创建数据集仓库: {repo_id}")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            exist_ok=True,
            private=False  # 设为True则为私有数据集
        )
        print("✅ 仓库创建成功")
    except Exception as e:
        print(f"仓库已存在或创建失败: {e}")
    
    # 4. 上传文件
    data_dir = Path("data")
    
    # 4.1 上传raw目录（原始数据）
    print("\n📤 上传raw目录...")
    api.upload_folder(
        folder_path=str(data_dir / "raw"),
        path_in_repo="raw",
        repo_id=repo_id,
        repo_type="dataset"
    )
    print("✅ raw目录上传完成")
    
    # 4.2 上传eval目录（评估数据）
    print("\n📤 上传eval目录...")
    api.upload_folder(
        folder_path=str(data_dir / "eval"),
        path_in_repo="eval",
        repo_id=repo_id,
        repo_type="dataset"
    )
    print("✅ eval目录上传完成")
    
    # 4.3 上传README
    print("\n📤 创建数据集卡片...")
    readme_content = """# Juris-RAG 数据集

## 数据集描述

Juris-RAG是一个中文法律检索增强生成（RAG）系统的数据集，包含：

- 中国法律法规文本
- CAIL司法案例数据集
- 评估数据集

## 数据集结构

```
raw/
  ├── criminal_code.txt      # 刑法
  ├── civil_code.txt         # 民法典
  ├── administrative_law.txt # 行政处罚法
  ├── labor_law.txt          # 劳动法
  ├── commercial_law.txt     # 公司法
  └── cail_cases.json        # CAIL案例数据（约140MB）

eval/
  └── eval_set.json          # 评估数据集
```

## 使用方法

### 方法1：使用huggingface_hub

```python
from huggingface_hub import hf_hub_download

# 下载特定文件
file_path = hf_hub_download(
    repo_id="YOUR_USERNAME/juris-rag-dataset",
    filename="raw/cail_cases.json",
    repo_type="dataset"
)
```

### 方法2：使用datasets库

```python
from datasets import load_dataset

dataset = load_dataset("YOUR_USERNAME/juris-rag-dataset")
```

### 方法3：批量下载

```bash
# 安装huggingface-cli
pip install huggingface_hub

# 下载整个数据集
huggingface-cli download YOUR_USERNAME/juris-rag-dataset --repo-type dataset --local-dir ./data
```

## 数据来源

- **法律文本**: 中国法律法规公开数据
- **CAIL案例**: [CAIL 2018数据集](https://github.com/thunlp/CAIL)

## 许可证

本数据集遵循原始数据的许可证：
- 法律文本：公开数据
- CAIL数据集：遵循其原始许可证

## 引用

如果使用本数据集，请引用：

```
@misc{juris-rag-dataset,
  title={Juris-RAG Dataset},
  author={Your Name},
  year={2026},
  url={https://huggingface.co/datasets/YOUR_USERNAME/juris-rag-dataset}
}
```
"""
    
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset"
    )
    print("✅ 数据集卡片创建完成")
    
    print(f"\n🎉 数据集上传完成！")
    print(f"访问地址: https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    upload_dataset()
