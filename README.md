# 🏛️ Juris-RAG：法律领域智能问答系统

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)](https://langchain.com)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 项目简介

**Juris-RAG** 是一个基于检索增强生成（RAG）技术的中文法律领域智能问答系统。该系统整合了刑法法条和司法案例数据，通过向量检索和大语言模型，为用户提供准确、专业、可追溯的法律咨询服务。

### ✨ 核心特性

| 特性 | 描述 |
|------|------|
| 📚 **领域知识库** | 整合刑法法条 + 5000+ CAIL司法案例 |
| 🔍 **语义检索** | 基于BGE-M3向量模型的精准语义匹配 |
| 🤖 **智能生成** | Qwen2.5-7B大模型生成专业回答 |
| 💬 **多轮对话** | 支持上下文理解，连续追问 |
| 📝 **引用追溯** | 每个回答标注信息来源 |
| 🚫 **拒绝不确定** | 对超出知识范围的问题明确告知 |
| 🌐 **Web交互** | Gradio构建的友好界面 |
| 📊 **自动评估** | 准确率、引用F1、幻觉率等指标 |

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      用户界面 (Gradio)                       │
├─────────────────────────────────────────────────────────────┤
│                     RAG引擎 (rag_engine.py)                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ 问题改写    │→│ 向量检索     │→│ 答案生成 + 引用      │ │
│  │ (多轮对话)  │  │ (Top-K)     │  │ (Qwen2.5)          │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                  向量数据库 (ChromaDB)                       │
│  ┌─────────────────────┐  ┌───────────────────────────────┐│
│  │ 刑法法条 (Statute)  │  │ CAIL案例 (Case) - 5000+条    ││
│  └─────────────────────┘  └───────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## 📁 项目结构

```
Juris-RAG/
├── 📄 app.py                 # Gradio Web应用入口
├── 📄 eval.py                # 评估脚本
├── 📄 requirements.txt       # 依赖列表
├── 📄 README.md              # 项目说明
├── 📄 .env                   # 环境变量配置（需自建）
├── 📄 .gitignore             # Git忽略配置
│
├── 📂 src/                   # 源代码目录
│   ├── __init__.py
│   ├── config.py             # 配置参数
│   ├── data_processing.py    # 数据处理与向量化
│   └── rag_engine.py         # RAG核心引擎
│
├── 📂 data/                  # 数据目录
│   ├── raw/                  # 原始数据
│   │   ├── criminal_code.txt # 刑法文本
│   │   └── cail_cases.json   # CAIL案例数据
│   ├── eval/                 # 评估数据
│   └── vector_db/            # 向量数据库（自动生成）
│
└── 📂 reports/               # 实验报告目录
    └── 学号-姓名-01-NLP.md   # 实验报告
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/your-username/Juris-RAG.git
cd Juris-RAG

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置API Key

创建 `.env` 文件：

```env
SILICONFLOW_API_KEY=your_api_key_here
```

或直接设置环境变量：

```bash
# Windows PowerShell
$env:SILICONFLOW_API_KEY="your_api_key_here"

# Windows CMD
set SILICONFLOW_API_KEY=your_api_key_here

# Linux/Mac
export SILICONFLOW_API_KEY=your_api_key_here
```

> 💡 API Key 获取：访问 [SiliconFlow](https://siliconflow.cn/) 注册并获取API Key

### 3. 准备数据

确保 `data/raw/` 目录下有以下文件：
- `criminal_code.txt` - 刑法文本
- `cail_cases.json` - CAIL案例数据（JSON Lines格式）

### 4. 构建向量数据库

```bash
python -m src.data_processing
```

输出示例：
```
📄 正在加载法条: ./data/raw/criminal_code.txt
✅ 加载法条完成，共 XXX 个文档块
⚖️ 正在加载 CAIL 案例: ./data/raw/cail_cases.json (限制 5000 条)
✅ 加载案例完成，共 5000 个文档

📊 数据集统计:
   总文档数: 5XXX
   按类型分布: {'statute': XXX, 'case': 5000}
   平均长度: XXX.X 字符

📦 准备向量化 5XXX 条文档...
✅ 向量数据库构建完成！已保存至 ./data/vector_db
```

### 5. 启动Web应用

```bash
python app.py
```

访问 http://localhost:7860 开始使用！

### 6. 运行评估

```bash
python eval.py
```

## 📊 评估指标

| 指标 | 说明 | 基线值 |
|------|------|--------|
| **Accuracy** | 回答正确率 | ~70% |
| **Citation F1** | 引用来源准确性 | ~80% |
| **Hallucination Rate** | 幻觉率（越低越好）| <10% |
| **Avg Latency** | 平均响应时间 | <3s |

## 🔧 配置说明

主要配置项在 `src/config.py`：

```python
# 模型配置
EMBEDDING_MODEL = "BAAI/bge-m3"      # 嵌入模型
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # 大语言模型

# RAG参数
CHUNK_SIZE = 500                      # 文本分块大小
RETRIEVAL_TOP_K = 5                   # 检索返回数量
LLM_TEMPERATURE = 0.1                 # 生成温度

# 长上下文支持
MAX_CONTEXT_LENGTH = 32768            # 最大上下文长度
MAX_HISTORY_TURNS = 10                # 保留的历史对话轮数
```

## 🎯 使用示例

### 基础问答

```
👤 用户: 故意杀人罪怎么判刑？

🤖 助手: 根据《刑法》第二百三十二条规定，故意杀人的，处死刑、无期徒刑或者
十年以上有期徒刑；情节较轻的，处三年以上十年以下有期徒刑。[来源1]

📚 引用来源:
[1] 中华人民共和国刑法 (statute)
    条款: 第二百三十二条
```

### 多轮对话

```
👤 用户: 盗窃罪怎么判？
🤖 助手: 根据《刑法》第二百六十四条...

👤 用户: 如果数额特别巨大呢？  # 引用上文
🤖 助手: 数额特别巨大或者有其他特别严重情节的...
```

### 拒绝不确定问题

```
👤 用户: 民法典关于合同的规定是什么？

🤖 助手: 抱歉，根据现有法律数据库，我无法准确回答此问题。
本系统主要涵盖刑法相关内容，民法典相关问题建议咨询专业律师。
```

## 📈 优化方向

- [ ] 扩展数据源（民法典、行政法等）
- [ ] 引入重排序模型提升检索质量
- [ ] LoRA微调提升领域准确率
- [ ] 添加更多评估基准（LegalBench等）
- [ ] 支持文档上传和实时索引

## 🛠️ 技术栈

| 组件 | 技术选型 |
|------|----------|
| 框架 | LangChain 0.1+ |
| 向量库 | ChromaDB |
| Embedding | BAAI/bge-m3 (via SiliconFlow) |
| LLM | Qwen/Qwen2.5-7B-Instruct |
| 前端 | Gradio 4.0+ |
| API | SiliconFlow |

## 📚 数据来源

1. **刑法法条**: 中华人民共和国刑法（2020修正）
2. **司法案例**: [CAIL2018](https://github.com/china-ai-law-challenge/CAIL2018) 中国法律智能挑战赛数据集

## ⚠️ 免责声明

本系统仅供学习和研究使用，不构成法律建议。如有实际法律问题，请咨询专业律师。

## 📄 许可证

MIT License

## 👨‍💻 作者

**[你的姓名]** - [学号]

---

> 📧 如有问题，请提交 Issue 或联系作者。
