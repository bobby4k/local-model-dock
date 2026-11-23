# Local Model Dock

Local Model Dock 旨在为本地 Agent 提供模型发现、选择和 OpenAI 兼容接口。目标平台为 Apple Silicon Mac（oMLX）和 Linux（Ollama）；不支持 Intel Mac。

> **当前状态：开发早期，尚不能运行模型。** 目前只有命令入口、配置读取和初步候选筛选逻辑。模型发现、下载、后端管理、API 网关、更新确认、回退和 Telegram 通知均未实现，请勿用于生产环境。

## 环境与安装

需要 Python 3.11 或更新版本。目前不需要安装 oMLX 或 Ollama：后端接入尚未实现。

```bash
git clone https://github.com/bobby4k/local-model-dock.git
cd local-model-dock
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

查看命令帮助：

```bash
local-model-dock --help
```

## 命令

计划支持以下命令；**目前均为占位入口**，执行时会输出 `not implemented` 并返回非零状态，不会启动服务、检查更新、下载模型或停止进程。

```bash
local-model-dock start
local-model-dock update
local-model-dock stop
local-model-dock restart
```

## 配置

可复制配置示例供后续开发使用：

```bash
cp .env.example .env
```

配置读取模块支持以下字段；**当前命令入口尚未加载或使用这些配置**，因此设置它们不会启用相应功能。环境变量在读取时优先于 `.env`。

| 字段 | 默认值 | 规划用途 |
| --- | --- | --- |
| `MODEL_BUDGET_GB` | `20` | 磁盘预算提醒阈值，必须是正数 |
| `AUTO_DOWNLOAD` | `true` | 首次下载自动模式，取值 `true` 或 `false` |
| `MODEL_FAMILIES` | `deepseek,qwen` | 逗号分隔的模型家族优先顺序 |
| `API_KEY` | 空 | 未来的网关访问密钥 |
| `TELEGRAM_BOT_TOKEN` | 空 | 可选的 Telegram Bot Token |
| `TELEGRAM_USER_ID` | 空 | 允许确认的 Telegram 用户 ID |
| `TELEGRAM_CHAT_ID` | 空 | 允许确认的 Telegram 聊天 ID |

不要将真实密钥提交到仓库；`.env` 已列入 `.gitignore`。API 认证和 Telegram 授权目前均未实现，配置密钥不会提供保护。

## 运行测试

在仓库根目录执行：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

测试覆盖当前的配置、占位命令和部分筛选规则，不代表模型运行、网络安全或真实硬件已验证。

## 设计与实施计划

- [设计文档](docs/superpowers/specs/2026-11-23-agent-model-dock-design.md)
- [实施计划](docs/superpowers/plans/2026-11-23-local-agent-model-plan.md)
