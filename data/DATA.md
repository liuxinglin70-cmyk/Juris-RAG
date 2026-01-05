# 数据文件说明

本项目需要以下数据文件，由于文件较大，未包含在 Git 仓库中。

## 📁 需要的数据文件

### 1. 法律文本文件

这些文件应放在 `data/raw/` 目录下：

- **刑法**: `criminal_code.txt` (已包含)
- **民法典**: `civil_code.txt` (已包含)
- **行政处罚法**: `administrative_law.txt` (已包含)
- **劳动法**: `labor_law.txt` (已包含)
- **公司法**: `commercial_law.txt` (已包含)

所有法律文本已包含在仓库中，可直接使用。

### 2. CAIL 司法案例数据集 ⚠️

**文件名**: `cail_cases.json`  
**大小**: ~140 MB  
**格式**: JSON数组  
**说明**: 中国法研杯（CAIL）司法人工智能挑战赛数据集

#### 获取方式

**选项1: 使用原始数据集**

从 CAIL 官方获取：
- 官网: https://cail.oss-cn-qingdao.aliyuncs.com/
- GitHub: https://github.com/thunlp/CAIL

下载后：
1. 解压缩
2. 重命名为 `cail_cases.json`
3. 放入 `data/raw/` 目录

**选项2: 使用预处理数据集**

如果你有访问权限，可以从以下位置下载预处理版本：
- 百度网盘: [链接]（提取码: xxxx）
- Google Drive: [链接]

**选项3: 使用小样本数据集（用于测试）**

创建一个小规模测试数据集：

```bash
# 从完整数据集中提取前100个案例（用于快速测试）
python -c "
import json
with open('data/raw/cail_cases.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
with open('data/raw/cail_cases_test.json', 'w', encoding='utf-8') as f:
    json.dump(data[:100], f, ensure_ascii=False, indent=2)
"
```

## 📋 数据格式

### CAIL 案例数据格式

```json
[
  {
    "fact": "案情描述...",
    "meta": {
      "accusation": ["故意伤害", "寻衅滋事"],
      "relevant_articles": [234, 293],
      "term_of_imprisonment": {
        "imprisonment": 36,
        "death_penalty": false,
        "life_imprisonment": false
      },
      "punish_of_money": 0
    }
  }
]
```

## ✅ 验证数据文件

运行验证脚本检查数据文件：

```bash
python verify_data.py
```

应该看到：
```
✅ criminal_code.txt: 通过
✅ civil_code.txt: 通过
✅ administrative_law.txt: 通过
✅ labor_law.txt: 通过
✅ commercial_law.txt: 通过
✅ cail_cases.json: 通过
```

## 🔄 处理数据

数据文件准备好后，运行数据处理脚本生成向量数据库：

```bash
python src/data_processing.py
```

这将：
1. 加载所有法律文本和案例
2. 进行文本分块和清洗
3. 生成向量embeddings
4. 创建 ChromaDB 向量数据库
5. 保存到 `data/vector_db/` 目录

## ⚙️ 配置选项

可以通过环境变量调整案例数量：

```bash
# Windows
set CAIL_CASE_LIMIT=10000

# Linux/Mac
export CAIL_CASE_LIMIT=10000
```

然后运行数据处理。

## 📊 数据统计

完整数据集统计：

| 数据类型 | 数量 | 大小 |
|---------|------|------|
| 刑法法条 | ~500 | ~220 KB |
| 民法典条文 | ~1,200 | ~350 KB |
| 行政法条文 | ~120 | ~33 KB |
| 劳动法条文 | ~100 | ~28 KB |
| 公司法条文 | ~400 | ~105 KB |
| CAIL案例 | 100,000+ | ~140 MB |

## 📝 注意事项

1. **大文件不在Git中**: `cail_cases.json` 因为超过 GitHub 的 100MB 限制，不包含在仓库中
2. **本地生成**: 向量数据库（`data/vector_db/`）也不在 Git 中，需要本地生成
3. **数据版权**: 使用 CAIL 数据集请遵守其使用协议
4. **磁盘空间**: 确保有足够空间（至少 500 MB）

## ❓ 常见问题

**Q: 没有 CAIL 数据集可以运行吗？**  
A: 可以。系统会自动跳过案例加载，只使用法条数据。但问答效果会受影响。

**Q: 如何减少数据量？**  
A: 设置环境变量 `CAIL_CASE_LIMIT=5000` 只加载 5000 个案例。

**Q: 数据处理要多久？**  
A: 取决于案例数量和网络速度，通常 10-30 分钟。

**Q: 可以使用自己的数据吗？**  
A: 可以。参照 CAIL 格式准备 JSON 文件即可。

## 🔗 相关链接

- CAIL 官网: https://cail.oss-cn-qingdao.aliyuncs.com/
- LawRefBook: https://github.com/RanKKI/LawRefBook
- 中国裁判文书网: https://wenshu.court.gov.cn/
