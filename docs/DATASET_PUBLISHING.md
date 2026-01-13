# 数据集发布指南

本文档说明如何将 Juris-RAG 数据集发布到开源平台。

## 📦 数据集概览

- **总大小**: ~1.9GB
- **raw/**: 140MB（法律文本 + CAIL案例）
- **vector_db/**: 1.8GB（向量数据库，不建议上传）

## 🎯 推荐方案

### 方案1: Hugging Face Hub（最推荐）⭐⭐⭐

**优点**：
- 专为ML数据集设计
- 免费无限存储
- 自动版本控制
- 良好的可发现性

**步骤**：

1. **安装依赖**
```bash
pip install huggingface_hub
```

2. **登录Hugging Face**
```bash
huggingface-cli login
```
或访问 https://huggingface.co/settings/tokens 获取token

3. **上传数据集**
```bash
python scripts/upload_to_huggingface.py
```

按提示输入你的Hugging Face用户名，脚本将自动：
- 创建数据集仓库
- 上传raw/和eval/目录
- 生成数据集卡片（README）

4. **数据集地址**
```
https://huggingface.co/datasets/your-username/juris-rag-dataset
```

### 方案2: GitHub + Git LFS（适合小规模）

**限制**：免费账户2GB存储 + 1GB/月带宽

**步骤**：

1. **安装Git LFS**
```bash
git lfs install
```

2. **配置追踪大文件**

创建 `.gitattributes`:
```
data/raw/*.json filter=lfs diff=lfs merge=lfs -text
data/raw/*.txt filter=lfs diff=lfs merge=lfs -text
```

3. **更新 .gitignore**

移除 `data/raw/*.json` 排除规则，但保留：
```
# 向量数据库不上传
data/vector_db/
```

4. **提交并推送**
```bash
git lfs track "data/raw/*.json"
git add .gitattributes data/raw/ data/eval/
git commit -m "Add dataset files with Git LFS"
git push
```

### 方案3: Zenodo（学术出版）

**优点**：
- 获得DOI，可被引用
- 永久存储
- 每个数据集50GB限制

**步骤**：
1. 访问 https://zenodo.org/
2. 创建账户并登录
3. 点击 "Upload" → "New upload"
4. 上传数据文件
5. 填写元数据（标题、作者、描述等）
6. 发布并获得DOI

### 方案4: 百度网盘/阿里云盘（国内备选）

适合国内用户快速下载，但不利于版本控制和国际传播。

**步骤**：
1. 压缩data/raw/目录
```bash
cd data
tar -czf juris-rag-data.tar.gz raw/ eval/
```
2. 上传到网盘
3. 在README中提供分享链接

## 📝 混合方案（最佳实践）

推荐组合使用：

| 数据类型 | 平台 | 原因 |
|---------|------|------|
| **原始数据**（raw/） | Hugging Face | 版本控制、易用性 |
| **代码** | GitHub | 代码托管、协作 |
| **向量数据库** | 本地生成 | 太大，用户自行构建 |

## 🔧 用户使用流程

### 使用Hugging Face数据集

用户只需运行：

```bash
# 克隆代码仓库
git clone https://github.com/your-username/Juris-RAG.git
cd Juris-RAG

# 安装依赖
pip install -r requirements.txt

# 自动下载数据集
python scripts/setup_data.py
```

系统会自动：
1. 从Hugging Face下载数据
2. 创建向量数据库目录
3. 首次运行时自动构建向量索引

## 📊 数据集元数据示例

在Hugging Face上传时，建议包含以下元数据：

```yaml
# dataset_info.yaml
dataset_info:
  description: 中文法律RAG数据集，包含多领域法律文本和CAIL案例
  citation: |
    @dataset{juris_rag_2026,
      title={Juris-RAG Dataset},
      author={Your Name},
      year={2026},
      url={https://huggingface.co/datasets/your-username/juris-rag-dataset}
    }
  homepage: https://github.com/your-username/Juris-RAG
  license: mit
  features:
    - name: legal_texts
      description: 中国法律法规文本
    - name: cail_cases
      description: CAIL司法案例数据
  size_categories:
    - 100M<n<1B
  language:
    - zh
  task_categories:
    - question-answering
    - text-retrieval
```

## 🚀 快速命令

### 上传到Hugging Face
```bash
python scripts/upload_to_huggingface.py
```

### 从Hugging Face下载
```bash
python scripts/download_from_huggingface.py
```

### 完整的用户设置
```bash
python scripts/setup_data.py
```

## ⚠️ 注意事项

1. **不要上传向量数据库**（vector_db/）：
   - 文件太大（1.8GB）
   - 不同环境可能不兼容
   - 用户可以快速重建

2. **数据许可证**：
   - 确保CAIL数据集使用符合其原始许可证
   - 法律文本为公开数据

3. **版本控制**：
   - 使用语义化版本号
   - 在README中记录数据变更

4. **数据隐私**：
   - CAIL案例已脱敏
   - 不包含敏感个人信息

## 📮 联系方式

如有问题，请：
- 提交 GitHub Issue
- 联系维护者

---

最后更新：2026-01-13
