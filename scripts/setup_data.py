"""
自动化数据设置脚本
用户首次使用项目时运行此脚本，自动从Hugging Face下载数据
"""
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

# 添加项目根目录到path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_data_exists():
    """检查数据是否已存在"""
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    eval_dir = data_dir / "eval"
    
    required_files = [
        raw_dir / "criminal_code.txt",
        raw_dir / "civil_code.txt",
        raw_dir / "cail_cases.json",
        eval_dir / "eval_set.json"
    ]
    
    missing_files = [f for f in required_files if not f.exists()]
    
    if not missing_files:
        print("✅ 所有数据文件已存在")
        return True
    else:
        print("⚠️  缺少以下文件:")
        for f in missing_files:
            print(f"  - {f.relative_to(project_root)}")
        return False

def download_data(repo_id: str):
    """从Hugging Face下载数据"""
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    
    print(f"\n📥 正在从 {repo_id} 下载数据...")
    print("这可能需要几分钟，请耐心等待...")
    print("\n提示: 如果是私有数据集或下载失败，可能需要登录")
    print("设置环境变量: set HF_TOKEN=your_token")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(data_dir),
            ignore_patterns=["*.md", ".gitattributes"],
            token=os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        )
        print("\n✅ 数据下载完成！")
        return True
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        if "401" in str(e) or "403" in str(e):
            print("\n可能是权限问题，请检查:")
            print("1. 数据集是否为私有？")
            print("2. 是否需要设置访问令牌？")
            print("   设置方法: set HF_TOKEN=hf_xxxxx")
        else:
            print("\n请检查:")
            print("1. 数据集ID是否正确")
            print("2. 网络连接是否正常")
        return False

def setup_vector_db():
    """创建向量数据库目录"""
    vector_db_dir = project_root / "data" / "vector_db"
    vector_db_dir.mkdir(exist_ok=True)
    
    # 创建各个法律领域的子目录
    for domain in ["criminal", "civil", "administrative", "labor", "commercial"]:
        (vector_db_dir / domain).mkdir(exist_ok=True)
    
    print("✅ 向量数据库目录已创建")
    print("提示: 首次运行应用时，系统将自动生成向量数据库")

def main():
    print("=" * 60)
    print("Juris-RAG 数据设置向导")
    print("=" * 60)
    
    # 检查数据是否存在
    if check_data_exists():
        choice = input("\n数据已存在，是否重新下载？(y/N): ")
        if choice.lower() != 'y':
            print("跳过数据下载")
            return
    
    # 获取数据集ID
    print("\n请提供Hugging Face数据集ID")
    print("格式: username/dataset-name")
    print("示例: yourusername/juris-rag-dataset")
    
    repo_id = input("\n数据集ID: ").strip()
    
    if not repo_id or '/' not in repo_id:
        print("❌ 无效的数据集ID")
        return
    
    # 下载数据
    if download_data(repo_id):
        # 设置向量数据库目录
        setup_vector_db()
        
        print("\n" + "=" * 60)
        print("🎉 数据设置完成！")
        print("=" * 60)
        print("\n下一步:")
        print("1. 运行应用: python app.py")
        print("2. 系统将自动构建向量数据库（首次运行需要一些时间）")
    else:
        print("\n" + "=" * 60)
        print("⚠️  数据设置未完成")
        print("=" * 60)
        print("\n请参考 data/DATA.md 获取手动下载数据的说明")

if __name__ == "__main__":
    main()
