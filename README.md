# LangChain 学习项目

这是我在学习 [LangChain](https://github.com/langchain-ai/langchain) 过程中编写的一些代码练习。该项目旨在帮助我更好地理解和掌握 LangChain 的使用方法，并结合 AI 工具进行开发实践。

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

本项目使用 `pyproject.toml` 来管理依赖。你可以通过以下命令安装依赖：

```bash
pip install -e .
```

## 🚀 快速开始

1. 克隆仓库：

   ```bash
   git clone https://github.com/supengxu/langchainStudy.git
   cd langchainStudy
   ```

2. 安装依赖：

   ```bash
   pip install -e .
   ```

3. 创建 `.env` 文件并填入你的 API 密钥等配置。

4. 运行测试脚本：

   ```bash
   python test.py
   ```

## 📁 目录结构

- `test.py`: 主要的 LangChain 测试代码。
- `.env`: 存放环境变量（本地使用，不提交到 Git）。
- `pyproject.toml`: Python 项目配置文件，包含依赖项。
- `.gitignore`: 指定 Git 忽略的文件和目录。

## 🧠 技术栈

- [LangChain](https://python.langchain.com/)
- [Tavily](https://tavily.com/)
- [LangSmith](https://www.langchain.com/langsmith)
- [Python-dotenv](https://github.com/theskumar/python-dotenv)

---

如果你有任何问题或建议，欢迎提 Issue 或 Pull Request！