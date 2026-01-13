"""
快速上传脚本 - 无需CLI登录
直接使用token上传数据到Hugging Face
"""
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, login

def main():
    print("=" * 60)
    print("Juris-RAG 数据集上传工具")
    print("=" * 60)
    
    # 1. 获取token
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    
    if not token:
        print("\n🔑 需要Hugging Face访问令牌")
        print("获取方式: https://huggingface.co/settings/tokens")
        print("\n选择登录方式:")
        print("1. 输入访问令牌（推荐）")
        print("2. 通过浏览器登录")
        
        choice = input("\n选择 (1/2): ").strip()
        
        if choice == "1":
            token = input("\n请粘贴你的访问令牌: ").strip()
            if not token:
                print("❌ 未提供令牌")
                return
            try:
                login(token=token, add_to_git_credential=True)
                print("✅ 登录成功！")
            except Exception as e:
                print(f"❌ 登录失败: {e}")
                return
        elif choice == "2":
            try:
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
    username = input("\n输入你的Hugging Face用户名: ").strip()
    if not username:
        print("❌ 用户名不能为空")
        return
    
    repo_id = f"{username}/juris-rag-dataset"
    
    api = HfApi(token=token if token else None)
    
    # 3. 创建数据集仓库
    print(f"\n📦 创建数据集仓库: {repo_id}")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            exist_ok=True,
            private=False,  # 设为True则为私有数据集
            token=token if token else None
        )
        print("✅ 仓库创建成功")
    except Exception as e:
        print(f"⚠️  仓库已存在或创建失败: {e}")
    
    # 4. 上传文件
    data_dir = Path("data")
    
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return
    
    # 4.1 上传raw目录（原始数据）
    raw_dir = data_dir / "raw"
    if raw_dir.exists():
        print("\n📤 上传raw目录...")
        try:
            api.upload_folder(
                folder_path=str(raw_dir),
                path_in_repo="raw",
                repo_id=repo_id,
                repo_type="dataset",
                token=token if token else None
            )
            print("✅ raw目录上传完成")
        except Exception as e:
            print(f"❌ 上传失败: {e}")
            return
    else:
        print(f"⚠️  raw目录不存在: {raw_dir}")
    
    # 4.2 上传eval目录（评估数据）
    eval_dir = data_dir / "eval"
    if eval_dir.exists():
        print("\n📤 上传eval目录...")
        try:
            api.upload_folder(
                folder_path=str(eval_dir),
                path_in_repo="eval",
                repo_id=repo_id,
                repo_type="dataset",
                token=token if token else None
            )
            print("✅ eval目录上传完成")
        except Exception as e:
            print(f"❌ 上传失败: {e}")
            return
    else:
        print(f"⚠️  eval目录不存在: {eval_dir}")
    
    # 4.3 上传README
    print("\n📤 创建数据集卡片...")
    readme_content = f"""# Juris-RAG 数据集

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
    repo_id="{repo_id}",
    filename="raw/cail_cases.json",
    repo_type="dataset"
)
```

### 方法2：使用datasets库

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}")
```

### 方法3：批量下载

```bash
# 安装huggingface-cli
pip install huggingface_hub

# 下载整个数据集（使用Python脚本）
python scripts/setup_data.py
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
@misc{{juris-rag-dataset,
  title={{Juris-RAG Dataset}},
  author={{{username}}},
  year={{2026}},
  url={{https://huggingface.co/datasets/{repo_id}}}
}}
```
"""
    
    try:
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=token if token else None
        )
        print("✅ 数据集卡片创建完成")
    except Exception as e:
        print(f"❌ 上传失败: {e}")
        return
    
    print("\n" + "=" * 60)
    print("🎉 数据集上传完成！")
    print("=" * 60)
    print(f"\n访问地址: https://huggingface.co/datasets/{repo_id}")
    print(f"\n用户现在可以使用以下命令下载数据:")
    print(f"  python scripts/setup_data.py")
    print(f"  输入数据集ID: {repo_id}")

if __name__ == "__main__":
    main()
