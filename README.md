# Local Model Dock

Local Model Dock 旨在为本地 Agent 提供模型发现、选择和 OpenAI 兼容接口。目标平台为 Apple Silicon Mac 和 Linux，均使用 Ollama；不支持 Intel Mac。

> **当前状态：开发早期，请勿用于生产环境。** 本机实测过一个固定版本的 Ollama 备用模型；通用模型导入与验证、自动更新、回退和 Telegram 通知尚未接通。仓库不包含模型文件或验证记录，新安装不会自动获得已验证模型。

## 环境与安装

需要 Python 3.11 或更新版本。可以使用初始化脚本安装对应后端并安装项目：

在项目根目录运行与你的 shell 对应的命令，初始化成功后会自动激活当前终端中的环境：

```fish
source scripts/init.fish
```

```bash
# Bash 或 Zsh
source scripts/init-env.sh
```

也可以单独运行 `bash scripts/init.sh`，但它不能激活调用它的终端。

Apple Silicon Mac 未安装 Ollama 时需要预先安装 Homebrew；脚本通过 Homebrew 的 `ollama` 公式安装命令行程序，不会启动或注册 Homebrew 服务。Linux 需要在已配置的 APT 软件源中提供 `ollama` 包，安装时可能需要 sudo；若没有该包，脚本会提示查看 [Ollama 官方 Linux 安装文档](https://docs.ollama.com/linux)，不会自动添加软件源。APT 包的来源取决于系统配置，包安装过程可能启动或配置系统服务。脚本不下载模型；已有 `ollama` 命令时不会重新安装，也不需要 Homebrew。

Homebrew 的 `brew services start ollama` 会启动服务并设置登录时启动；本项目的 supervisor 则自行启动和管理 `ollama serve`，不要同时在同一端口运行另一个 Ollama 服务。

脚本会创建或复用 `.venv`，安装项目，并在 `.env` 不存在时从 `.env.example` 创建默认配置。`API_KEY` 缺失或为空时自动生成并写入 `.env`（权限设为 600），已有密钥和其他配置保留。如果只运行了 `bash scripts/init.sh`，请自行激活：fish 使用 `source .venv/bin/activate.fish`，bash/zsh 使用 `source .venv/bin/activate`。也可以手动安装项目：

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

`update` 默认只读检查 Hugging Face 的部分文本生成模型，并通过运行中的 supervisor 复核已下载模型的验证记录。只有仓库、版本、后端及其当前版本均与验证身份一致，且满足 MoE、访问、格式和资源等条件时才会显示推荐；其余结果仍是待验证候选。许可证元数据的推荐白名单目前仅含 `apache-2.0` 和 `mit`，不代表已完成法律审查，使用前仍需自行审阅。检查不会下载或切换模型；supervisor 不可用或身份无法确认时不会推荐：

```bash
local-model-dock update
```

每个可能推荐的候选需要两次身份复核；复核会读取和校验本地模型文件，可能耗时，且不保证运行时上下文内存估算准确。

显式下载功能仍处于开发中，**不是可用的通用 Ollama 安装流程**。下载前必须在交互终端输入完整仓库和不可变版本，超预算需再次确认；请自行审阅许可证、资源估算和其他风险：

```bash
local-model-dock update --install owner/model
local-model-dock update --install owner/model --activate
```

目前 Ollama 验证和启用仅针对固定来源和版本的模型，不支持任意下载的模型。模型 ID 使用模型名称和量化标签，启用状态保存在私有运行目录中，重启后仍会重新检查验证记录。supervisor 必须运行；验证会执行真实工具调用：

```bash
local-model-dock update --verify deepseek-v4-pro-qwen3.5-9b-uncensored:q4_k_m
local-model-dock update --verify deepseek-v4-pro-qwen3.5-9b-uncensored:q4_k_m --activate
local-model-dock update --enable deepseek-v4-pro-qwen3.5-9b-uncensored:q4_k_m
```

这些命令不会下载模型。验证或复核失败不会启用；已启用的固定模型在重启后通过复核可自动恢复。修改回退 URL 或量化设置不会自动下载、导入或替换当前模型。

在项目目录设置非空 `API_KEY` 后，可管理独立的 supervisor：

```bash
local-model-dock start
local-model-dock stop
local-model-dock restart
```

`start` 等待后端监听及网关绑定成功，并输出本机 API 地址；服务已经运行时会输出其实际地址。`restart` 重启后同样输出地址；升级前已运行的旧版 supervisor 可能需要先执行一次 `restart` 才支持重复 `start` 时查询地址。supervisor 持有自己启动的后端子进程；`stop` 通过 `.local-model-dock/control.sock` 请求其退出，不读取 PID 文件，也不向其他进程发信号。`restart` 先停止再启动；未运行时停止或重启会报错。各命令须在同一项目目录执行，运行目录必须由当前用户拥有且权限为 700。异常退出留下的旧 socket 会在持有独占锁后清理；非 socket 文件不会被删除。

Supervisor 是本项目的 Python 模块 `src/local_model_dock/supervisor.py`，使用项目 `.venv` 运行，并非独立安装或开机自启的系统服务。控制 socket、模型和运行记录保存在项目目录下的 `.local-model-dock/`。

就绪检测目前只检查后端 TCP 监听，不验证推理功能。若 supervisor 被强制终止（如 SIGKILL），后端子进程可能遗留；不要用 PID 文件清理其他进程，应先由用户核实进程归属。后端不响应终止时，supervisor 会保留锁并等待，不会强杀或启动替代进程。

网关默认监听 `0.0.0.0:8399`，后端仅监听 `127.0.0.1:11434`；可通过环境变量 `DOCK_GATEWAY_PORT` 和 `DOCK_BACKEND_PORT` 修改端口。网关需要 `Authorization: Bearer <API_KEY>`，支持 `/v1/models` 和 `/v1/chat/completions` 子集。未启用有效验证记录时模型列表为空，聊天请求返回 503。不要依赖 API 密钥作为网络隔离：防火墙应限制网关访问，后端回环端口仍可由本机其他用户访问。HTTP 不加密密钥或内容，需要加密时请配置 HTTPS；不要直接暴露到公网。SSD 缓存可能包含对话内容。

客户端的 OpenAI 兼容 Base URL 为 `http://<主机局域网 IP>:8399/v1`，模型名为 `local-model`，API Key 使用项目 `.env` 中的 `API_KEY`。本机可在激活虚拟环境后检查：

```bash
python -c 'from local_model_dock.config import load_config; import urllib.request; r=urllib.request.Request("http://127.0.0.1:8399/v1/models", headers={"Authorization": "Bearer " + load_config().api_key}); print(urllib.request.urlopen(r).read().decode())'
```

## 配置

初始化脚本会在 `.env` 不存在时创建它；手动安装时可复制配置示例（不要覆盖已有 `.env`）：

```bash
cp -n .env.example .env
```

配置读取模块支持以下字段；下载预算和家族顺序用于 `update`，`API_KEY` 用于网关，其余功能尚未接入生命周期命令。环境变量在读取时优先于 `.env`。

| 字段 | 默认值 | 规划用途 |
| --- | --- | --- |
| `MODEL_BUDGET_GB` | `50` | 磁盘预算提醒阈值，必须是正数 |
| `AUTO_DOWNLOAD` | `true` | 首次下载自动模式，取值 `true` 或 `false` |
| `MODEL_FAMILIES` | `deepseek,qwen` | 逗号分隔的模型家族优先顺序 |
| `FALLBACK_MODEL_URL` | Hugging Face 回退模型地址 | 可配置回退目标；自动回退尚未接入 |
| `FALLBACK_QUANTIZATION` | 回退 GGUF 量化版本 | 默认 `i1-Q4_K_M`；可在 `.env` 中修改，缺失或不唯一时明确报错 |
| `API_KEY` | 空 | 网关访问密钥；启动必填 |
| `TELEGRAM_BOT_TOKEN` | 空 | 可选的 Telegram Bot Token |
| `TELEGRAM_USER_ID` | 空 | 允许确认的 Telegram 用户 ID |
| `TELEGRAM_CHAT_ID` | 空 | 允许确认的 Telegram 聊天 ID |

不要将真实密钥提交到仓库；`.env` 已列入 `.gitignore`。Telegram 授权目前未实现。

## 运行测试

在仓库根目录执行：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

测试覆盖配置、模拟后端和跨进程生命周期等，不代表所有模型、网络安全或硬件均已验证。

## 设计与实施计划

- [设计文档](docs/superpowers/specs/2026-11-23-agent-model-dock-design.md)
- [实施计划](docs/superpowers/plans/2026-11-23-local-agent-model-plan.md)
