# 大模型学习吧

这是我在学习大模型技术（如 [LangChain](https://github.com/langchain-ai/langchain)、[LlamaIndex](https://github.com/jerryjliu/gpt_index) 等）过程中编写的一些代码练习。该项目旨在帮助我更好地理解和掌握大模型应用开发实践。

## ⚠️ 环境变量说明

本项目使用了 `.env` 文件来管理敏感信息和环境配置。你需要在项目根目录下创建 `.env` 文件并填写以下内容：

```env
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_PROJECT="test-001"
TAVILY_API_KEY="your_tavily_api_key"
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="your_langsmith_api_key"
MODEL="ep-20250226110219-c9gkg"
OPENAI_API_KEY="your_openai_api_key"
BASE_URL="https://ark.cn-beijing.volces.com/api/v3"
```

请确保不要将 `.env` 文件提交到版本控制中（已加入 `.gitignore`）。

## 📦 依赖管理

本项目采用模块化结构，每个子目录可以拥有独立的依赖管理。你可以使用对应目录下的 `pyproject.toml` 安装模块特定依赖。

例如安装 LangChain 基础模块依赖：

```bash
pip install -e langchain/basic/
```

## 🚀 快速开始

1. 克隆仓库：

   ```bash
   git clone https://github.com/supengxu/langchainStudy.git
   cd langchainStudy
   ```

2. 安装全局依赖（可选）：

   ```bash
   pip install -e .
   ```

3. 创建 `.env` 文件并填入你的 API 密钥等配置。

4. 运行测试脚本（根据具体模块路径调整）：

   ```bash
   python langchain/basic/test.py
   ```

## 📁 目录结构

- `.env`: 存放环境变量（本地使用，不提交到 Git）。
- `pyproject.toml`: 根目录配置文件（可选用途）。
- `.gitignore`: 指定 Git 忽略的文件和目录。
- `README.md`: 项目说明文档。
- `langchain/`: LangChain 学习内容。
  - `basic/`: LangChain 基础知识相关代码。
  - `advanced/`: LangChain 进阶知识相关代码。
  - `agents/`: LangChain Agent 应用相关代码。
- `model-tuning/`: 大模型调优相关内容。
- `transformer-theory/`: Transformer 模型理论与实现。
- `projects/`: 大模型实战项目代码。

## 🧠 技术栈

- [LangChain](https://python.langchain.com/)
- [Tavily](https://tavily.com/)
- [LangSmith](https://www.langchain.com/langsmith)
- [Python-dotenv](https://github.com/theskumar/python-dotenv)
- [LlamaIndex](https://gpt-index.readthedocs.io/) (未来可能添加)
- [Transformers](https://huggingface.co/docs/transformers)
- [PyTorch](https://pytorch.org/)

---

如果你有任何问题或建议，欢迎提 Issue 或 Pull Request！