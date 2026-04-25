# patent_agent

一个“专利挖掘”动态对话智能体项目：输入任意发明点子后，系统会在最多 12 轮内动态追问并补齐信息，最终输出 Markdown 结构化专利交底书。

## 1. 需求对照与实现说明

### 1.1 意图识别与通用启动

- 支持任意自然语言输入作为起点（无需固定模板）。
- 首条消息进入统一挖掘流程：更新维度状态 -> 生成下一问或收敛报告。

### 1.2 动态追问与状态管理（核心）

- 固定追踪 10 个维度：`技术实现`、`创新点`、`应用场景`、`用户群体`、`性能参数`、`材料选择`、`成本控制`、`现有技术差异`、`潜在扩展功能`、`使用限制`。
- 每轮由模型解析用户输入，允许“一次回答多个维度”并批量更新。
- 每轮仅追问 1-2 个缺失维度，避免机械问答。
- 当用户说“我不知道/没想好”等，智能体会优先给出 2-3 个启发选项继续推进。
- 满足“全维度覆盖”或“达到轮数上限”即主动结束并生成报告。

### 1.3 结构化交底书生成

结束对话时输出 Markdown 交底书，包含：

1. 发明名称与摘要
2. 核心技术方案（含与现有技术差异）
3. 创新点与商业价值（融合用户群体/场景等）
4. 待补充技术细节（未完善维度及原因）

## 2. 架构设计

### 2.1 模块划分

- `main.py`：Gradio Web UI 入口（聊天区、状态栏、维度摘要、报告展示）。
- `demo_cli.py`：终端演示入口（便于录制演示视频）。
- `agent.py`：状态管理、轮次控制、维度合并、调用模型、收敛与报告生成。
- `prompts.py`：核心 System Prompt + 状态更新 Prompt + 追问 Prompt。

### 2.2 状态管理方式

`AgentState`（Pydantic）维护完整会话状态：

- `dimensions: Dict[str, DimensionItem]`
  - `status`: `missing | partial | covered`
  - `content`: 当前维度已确认信息
- `conversation_history`: 历史对话
- `current_round`: 当前轮次
- `max_rounds`: 最大轮次（默认 12）
- `finished`: 是否已结束

处理流程：

1. 用户输入进入 `handle_user_message`。
2. 调用状态解析 Prompt，提取本轮覆盖的维度更新。
3. 合并到状态（防止已覆盖维度退化）。
4. 计算是否收敛：全覆盖或达到最大轮数。
5. 未收敛则继续追问，收敛则产出最终报告。

## 3. 核心 Prompt

本项目核心提示词存放在 `prompts.py`，其中：

- `SYSTEM_PROMPT`：定义角色、10 维度采集规则、收敛规则、输出格式。
- `STATE_UPDATE_PROMPT`：强制模型输出 JSON，用于维度状态更新。
- `FOLLOW_UP_PROMPT`：控制每轮只问 1-2 个维度，并在 unknown 时给启发选项。

> 你可直接查看 `prompts.py` 获取完整 Prompt 全文。

## 4. 模型与框架

- 语言：Python
- 框架：Gradio（Web UI）
- LLM SDK：`openai`（使用 OpenAI 兼容接口）
- 默认模型：`deepseek-chat`
- 默认 Base URL：`https://api.deepseek.com/v1`

## 5. 安装与运行

### 5.1 安装依赖

```bash
pip install -r requirements.txt
```

### 5.2 配置环境变量

```bash
copy .env.example .env
```

在 `.env` 中填写真实密钥：

```env
OPENAI_API_KEY=sk-你的deepseek真实key
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-chat
OPENAI_TEMPERATURE=0.3
MAX_ROUNDS=12
```

也支持使用 `DEEPSEEK_API_KEY` 代替 `OPENAI_API_KEY`。

### 5.3 运行方式

Web UI：

```bash
python main.py
```

CLI 演示版（推荐用于录屏）：

```bash
python demo_cli.py
```

## 6. 测试用例演示（含“不知道”分支）

建议按以下输入顺序演示：

1. `我想发明一个能自动分类收纳衣物的衣柜。`
2. 在第 2~3 轮刻意输入：`我不知道怎么设计材料和成本。`
3. 观察系统输出 2-3 个启发式选项，并继续选择其一补充。
4. 多轮后观察系统自动收敛并输出完整 Markdown 交底书。

预期行为：

- 系统不会卡死；
- unknown 场景有启发式引导；
- 最终自动生成结构化报告。

## 7. 交付清单建议

根据题目提交时建议包含：

1. 演示视频（建议录制 `demo_cli.py` 或 `main.py` 全流程）。
2. 源码压缩包（本仓库全部文件）。
3. 本 README（含架构说明、状态管理说明、Prompt 说明、LLM 与框架说明）。

## 8. 项目结构

```text
patent_agent/
├── main.py          # Gradio Web UI 入口
├── demo_cli.py      # 终端演示入口（便于录屏）
├── agent.py         # 核心对话引擎与状态管理
├── prompts.py       # 核心 Prompt 模板
├── .env.example     # 环境变量示例
├── requirements.txt # 依赖清单
└── README.md        # 项目说明文档
```
