# 如何上传数据集到Hugging Face

## 问题：huggingface-cli 命令找不到

如果你遇到 `huggingface-cli: 无法将"huggingface-cli"项识别为 cmdlet` 错误，这是因为Python Scripts目录不在PATH中。

## ✅ 解决方案：使用Python脚本直接上传

我们提供了一个新的脚本 `quick_upload.py`，无需使用CLI命令。

### 步骤1：获取Hugging Face访问令牌

1. 访问 https://huggingface.co/settings/tokens
2. 点击 "New token"
3. 选择 "Write" 权限
4. 复制生成的令牌（格式：`hf_xxxxxxxxxxxxx`）

### 步骤2：运行上传脚本

```powershell
cd C:\Users\NUAA\Desktop\Juris-RAG
python scripts\quick_upload.py
```

### 步骤3：按提示操作

脚本会引导你：
1. **选择登录方式**：
   - 方式1（推荐）：直接粘贴令牌
   - 方式2：通过浏览器登录（可能不可用）

2. **输入用户名**：你的Hugging Face用户名

3. **自动上传**：脚本会自动上传所有数据

## 🎯 完整示例

```powershell
(base) PS C:\Users\NUAA\Desktop\Juris-RAG> python scripts\quick_upload.py

============================================================
Juris-RAG 数据集上传工具
============================================================

🔑 需要Hugging Face访问令牌
获取方式: https://huggingface.co/settings/tokens

选择登录方式:
1. 输入访问令牌（推荐）
2. 通过浏览器登录

选择 (1/2): 1

请粘贴你的访问令牌: hf_xxxxxxxxxxxxx
✅ 登录成功！

输入你的Hugging Face用户名: yourusername

📦 创建数据集仓库: yourusername/juris-rag-dataset
✅ 仓库创建成功

📤 上传raw目录...
✅ raw目录上传完成

📤 上传eval目录...
✅ eval目录上传完成

📤 创建数据集卡片...
✅ 数据集卡片创建完成

============================================================
🎉 数据集上传完成！
============================================================

访问地址: https://huggingface.co/datasets/yourusername/juris-rag-dataset
```

## 🔧 其他方法

### 方法A：设置环境变量（一劳永逸）

```powershell
# 设置环境变量
setx HF_TOKEN "hf_xxxxxxxxxxxxx"

# 重新打开PowerShell，然后直接运行
python scripts\quick_upload.py
```

### 方法B：找到huggingface-cli的完整路径

```powershell
# 查找huggingface-cli.exe
where.exe /R C:\Users\NUAA\AppData huggingface-cli.exe

# 使用完整路径运行
C:\Users\NUAA\AppData\Roaming\Python\Python313\Scripts\huggingface-cli.exe login
```

### 方法C：添加Scripts到PATH

1. 找到Python Scripts目录：
   ```powershell
   python -c "import sys; print(sys.prefix + '\\Scripts')"
   ```

2. 将该路径添加到系统PATH环境变量

3. 重新打开PowerShell

## 📊 上传进度监控

上传大文件（如140MB的cail_cases.json）可能需要几分钟。脚本会显示进度：

```
📤 上传raw目录...
Uploading files: 100%|████████████████| 7/7 [02:15<00:00, 19.32s/file]
✅ raw目录上传完成
```

## ⚠️ 常见问题

### Q1: 上传失败，提示401/403错误
**A**: 检查令牌权限，确保选择了"Write"权限

### Q2: 网络连接超时
**A**: 检查网络，或使用代理：
```powershell
$env:HTTP_PROXY="http://proxy:port"
$env:HTTPS_PROXY="http://proxy:port"
python scripts\quick_upload.py
```

### Q3: 文件太大无法上传
**A**: Hugging Face支持大文件，但需要稳定网络。可以分步上传：
```python
# 修改quick_upload.py，只上传特定目录
# 注释掉不需要的上传部分
```

## 💡 推荐工作流

1. **本地测试**: 确保所有数据在data/目录下
2. **运行上传**: `python scripts\quick_upload.py`
3. **验证**: 访问Hugging Face查看上传结果
4. **更新README**: 将数据集链接添加到项目README

## 🔗 相关链接

- Hugging Face令牌: https://huggingface.co/settings/tokens
- 数据集文档: https://huggingface.co/docs/hub/datasets
- 上传大文件: https://huggingface.co/docs/hub/repositories-getting-started

---

最后更新：2026-01-13
