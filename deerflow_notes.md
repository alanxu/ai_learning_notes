# DeerFlow 2.0 - Deep Dive Architecture Notes

> **DeerFlow** (**D**eep **E**xploration and **E**fficient **R**esearch **Flow**) is an open-source **super agent harness** by ByteDance that orchestrates sub-agents, memory, and sandboxes -- powered by extensible skills. Version 2.0 is a ground-up rewrite sharing no code with v1. Built on **LangGraph** and **LangChain**, it is model-agnostic (any OpenAI-compatible LLM).
>
> GitHub: [bytedance/deer-flow](https://github.com/bytedance/deer-flow)

---

## Table of Contents

- [1. System Architecture](#1-system-architecture)
  - [1.1 Service Topology](#11-service-topology)
  - [1.2 Harness / App Split](#12-harness--app-split)
  - [1.3 Request Flow](#13-request-flow)
- [2. Agent System](#2-agent-system)
  - [2.1 Agent Factory](#21-agent-factory)
  - [2.2 Lead Agent](#22-lead-agent)
  - [2.3 System Prompt Engineering](#23-system-prompt-engineering)
  - [2.4 Thread State](#24-thread-state)
  - [2.5 Runtime Features](#25-runtime-features)
- [3. Middleware Chain](#3-middleware-chain)
  - [3.1 Execution Model](#31-execution-model)
  - [3.2 Middleware Catalog](#32-middleware-catalog)
  - [3.3 Hard Ordering Constraints](#33-hard-ordering-constraints)
- [4. Sandbox System](#4-sandbox-system)
  - [4.1 Three Sandbox Modes](#41-three-sandbox-modes)
  - [4.2 Sandbox Provider Pattern](#42-sandbox-provider-pattern)
  - [4.3 Local Sandbox](#43-local-sandbox)
  - [4.4 AIO Sandbox (Docker)](#44-aio-sandbox-docker)
  - [4.5 Provisioner Sandbox (Kubernetes)](#45-provisioner-sandbox-kubernetes)
  - [4.6 Docker-outside-of-Docker (DooD)](#46-docker-outside-of-docker-dood)
  - [4.7 Virtual Path Mapping](#47-virtual-path-mapping)
  - [4.8 Sandbox Tools](#48-sandbox-tools)
- [5. Model / LLM Provider System](#5-model--llm-provider-system)
  - [5.1 Model Factory](#51-model-factory)
  - [5.2 Supported Providers](#52-supported-providers)
  - [5.3 Claude Provider](#53-claude-provider)
- [6. Tool System](#6-tool-system)
  - [6.1 Tool Assembly](#61-tool-assembly)
  - [6.2 Tool Search (Deferred Loading)](#62-tool-search-deferred-loading)
  - [6.3 Community Tool Integrations](#63-community-tool-integrations)
- [7. Skills System](#7-skills-system)
  - [7.1 Skill Structure](#71-skill-structure)
  - [7.2 Built-in Skills](#72-built-in-skills)
  - [7.3 Progressive Loading](#73-progressive-loading)
- [8. Sub-Agent System](#8-sub-agent-system)
- [9. Memory System](#9-memory-system)
  - [9.1 Architecture](#91-architecture)
  - [9.2 Data Structure](#92-data-structure)
  - [9.3 Update Pipeline](#93-update-pipeline)
- [10. MCP Integration](#10-mcp-integration)
- [11. Guardrails System](#11-guardrails-system)
- [12. Gateway API](#12-gateway-api)
- [13. Frontend Architecture](#13-frontend-architecture)
  - [13.1 Tech Stack](#131-tech-stack)
  - [13.2 App Router Structure](#132-app-router-structure)
  - [13.3 Core Business Logic](#133-core-business-logic)
  - [13.4 Data Flow Pattern](#134-data-flow-pattern)
  - [13.5 Chat Modes](#135-chat-modes)
- [14. Configuration System](#14-configuration-system)
- [15. Docker Infrastructure & Deployment](#15-docker-infrastructure--deployment)
  - [15.1 Development Commands](#151-development-commands)
  - [15.2 Production Compose](#152-production-compose)
  - [15.3 Development Compose](#153-development-compose)
  - [15.4 Nginx Routing](#154-nginx-routing)
- [16. Runtime System](#16-runtime-system)
  - [16.1 Run Manager](#161-run-manager)
  - [16.2 Stream Bridge](#162-stream-bridge)
- [17. IM Channels](#17-im-channels)
- [18. Design Patterns & Principles](#18-design-patterns--principles)

---

## 1. System Architecture

### 1.1 Service Topology

DeerFlow runs as four main services behind an Nginx reverse proxy:

```
                        ┌─────────────────────────────────────────────┐
                        │                  Nginx (:2026)              │
                        │            Unified Entry Point              │
                        └──────┬──────────┬──────────────┬────────────┘
                               │          │              │
                    /api/langgraph/*   /api/*          /*
                               │          │              │
                        ┌──────▼──┐  ┌────▼────┐  ┌─────▼─────┐
                        │LangGraph│  │ Gateway │  │ Frontend  │
                        │ (:2024) │  │ (:8001) │  │  (:3000)  │
                        └─────────┘  └─────────┘  └───────────┘
                             │
                     ┌───────┴────────┐
                     │  Agent Runtime │
                     │  Middlewares   │
                     │  Tools/Sandbox │
                     └────────────────┘
```

| Service | Port | Technology | Role |
|---------|------|------------|------|
| **Nginx** | 2026 | nginx:alpine | Reverse proxy, CORS, SSE support |
| **LangGraph Server** | 2024 | LangGraph SDK | Core agent runtime -- threads, streaming, middleware, tools |
| **Gateway API** | 8001 | FastAPI (uvicorn) | REST API -- models, MCP, skills, uploads, artifacts, memory |
| **Frontend** | 3000 | Next.js 16 / React 19 | Web interface |
| **Provisioner** (optional) | 8002 | FastAPI | K8s sandbox pod manager (provisioner mode only) |

### 1.2 Harness / App Split

The backend enforces a strict **one-way dependency boundary** (validated in CI by `tests/test_harness_boundary.py`):

```
┌──────────────────────────────────────────────┐
│  App Layer (app/)                             │
│  - FastAPI Gateway                            │
│  - IM Channels (Feishu, Slack, Telegram, etc)│
│  - Import prefix: app.*                       │
│  ──────────── can import ────────────▼        │
│  Harness Layer (packages/harness/deerflow/)   │
│  - Agent orchestration, tools, sandbox        │
│  - Models, MCP, skills, config, memory        │
│  - Published as: deerflow-harness             │
│  - Import prefix: deerflow.*                  │
│  - NEVER imports app.*                        │
└──────────────────────────────────────────────┘
```

The harness is designed to be publishable as an independent package (`deerflow-harness`), usable without the app layer.

### 1.3 Request Flow

```
User -> Frontend -> Nginx -> LangGraph Server
                                  │
                          make_lead_agent()
                                  │
                    ┌─────────────▼──────────────┐
                    │        Agent Loop           │
                    │  Model -> Middleware Chain   │
                    │    -> Tool Node -> Model     │
                    └─────────────────────────────┘
                                  │
                    Tools use SandboxProvider
                    for isolated code execution
```

---

## 2. Agent System

### 2.1 Agent Factory

Two entry points exist:

**`make_lead_agent(config: RunnableConfig)`** -- the application-level factory registered in `langgraph.json`. Config-driven, reads from `config.yaml`.

**`create_deerflow_agent()`** -- SDK-level, config-free factory for embedded use. Accepts plain Python arguments: `model`, `tools`, `system_prompt`, `middleware`, `features`, `extra_middleware`, `plan_mode`, `state_schema`, `checkpointer`, `name`.

### 2.2 Lead Agent

`make_lead_agent()` follows this sequence:

1. **Resolve model** -- request override > agent config > global default
2. **Create chat model** -- via `create_chat_model()` (dynamic provider resolution)
3. **Load tools** -- via `get_available_tools()` (config + MCP + built-in + subagent)
4. **Generate system prompt** -- via `apply_prompt_template()` (skills, memory, subagent instructions injected)
5. **Build middleware chain** -- via `_build_middlewares()` (14 middleware slots)
6. **Create agent** -- `create_agent()` with `ThreadState` as state schema

### 2.3 System Prompt Engineering

`apply_prompt_template()` composes the system prompt from template placeholders:

| Placeholder | Content |
|-------------|---------|
| `{agent_name}` | "DeerFlow 2.0" or custom agent name |
| `{soul}` | Loaded from agent's `SOUL.md` personality file |
| `{memory_context}` | Memory data in `<memory>` tags |
| `{skills_section}` | Available skills with progressive loading instructions |
| `{deferred_tools_section}` | Tool search available tools |
| `{subagent_section}` | Orchestration instructions with concurrency limits |
| `{acp_section}` | ACP agent integration mounts |

The prompt defines the agent's workflow: **CLARIFY -> PLAN -> ACT**, along with file management (virtual paths), citation format, and behavioral constraints.

### 2.4 Thread State

`ThreadState` extends LangGraph's `AgentState` with:

```python
class ThreadState(AgentState):
    sandbox: SandboxState        # {sandbox_id}
    thread_data: ThreadDataState # {workspace_path, uploads_path, outputs_path}
    title: str | None
    artifacts: list[str]         # deduplicated via merge_artifacts reducer
    todos: list | None
    uploaded_files: list[dict] | None
    viewed_images: dict[str, ViewedImageData]  # merge_viewed_images reducer
```

### 2.5 Runtime Features

`RuntimeFeatures` is a declarative dataclass for feature flags:

```python
@dataclass
class RuntimeFeatures:
    sandbox: bool | AgentMiddleware = True
    memory: bool | AgentMiddleware = True
    summarization: bool | AgentMiddleware = True
    subagent: bool | AgentMiddleware = True
    vision: bool | AgentMiddleware = True
    auto_title: bool | AgentMiddleware = True
    guardrail: bool | AgentMiddleware = True
```

Each accepts `True` (built-in default), `False` (disabled), or a custom `AgentMiddleware` instance. Middleware positioning is controlled via `@Next(anchor)` / `@Prev(anchor)` decorators.

---

## 3. Middleware Chain

### 3.1 Execution Model

The middleware chain is a **pipeline** (not an onion model). Most middlewares hook into a single lifecycle point:

- `before_*` hooks run in **forward order** (0 to N)
- `after_*` hooks run in **reverse order** (N to 0)

Available hooks: `before_agent`, `before_model`, `wrap_model_call`, `after_model`, `wrap_tool_call`, `after_agent`.

### 3.2 Middleware Catalog

| # | Middleware | Hook | Lead | Sub | Purpose |
|---|-----------|------|:----:|:---:|---------|
| 0 | **ThreadDataMiddleware** | `before_agent` | Y | Y | Creates per-thread directories (workspace, uploads, outputs) |
| 1 | **UploadsMiddleware** | `before_agent` | Y | N | Scans uploaded files, injects `<uploaded_files>` context |
| 2 | **SandboxMiddleware** | `before_agent` + `after_agent` | Y | Y | Lazy/eager sandbox acquisition and release |
| 3 | **DanglingToolCallMiddleware** | `wrap_model_call` | Y | N | Injects synthetic error responses for orphaned tool calls |
| 4 | **LLMErrorHandlingMiddleware** | `wrap_model_call` | Y | Y | Retry/backoff for transient LLM errors (429, 5xx) |
| 5 | **GuardrailMiddleware** | `wrap_tool_call` | Y | Y | Pre-tool-call authorization via pluggable providers |
| 6 | **SandboxAuditMiddleware** | `wrap_tool_call` | Y | Y | Classifies bash commands (high-risk/medium/pass), writes audit logs |
| 7 | **ToolErrorHandlingMiddleware** | `wrap_tool_call` | Y | Y | Catches tool exceptions, returns error ToolMessages |
| 8 | **DeferredToolFilterMiddleware** | `wrap_model_call` | Y | N | Removes deferred tool schemas to save context tokens |
| 9 | **SummarizationMiddleware** | `after_model` | Y | N | Context reduction approaching token limits |
| 10 | **TodoMiddleware** | `before_model` | Y | N | Plan mode task tracking with `write_todos` tool |
| 11 | **TokenUsageMiddleware** | `after_model` | Y | N | Logs input/output/total token counts |
| 12 | **TitleMiddleware** | `after_model` | Y | N | Auto-generates thread title after first exchange |
| 13 | **MemoryMiddleware** | `after_agent` | Y | N | Queues conversation for background memory extraction |
| 14 | **ViewImageMiddleware** | `before_model` | Y | N | Injects base64 image data for vision analysis |
| 15 | **SubagentLimitMiddleware** | `after_model` | Y | N | Caps concurrent `task` tool calls (clamped to [2,4]) |
| 16 | **LoopDetectionMiddleware** | `after_model` | Y | N | Detects repeated tool calls; warns at 3, stops at 5 |
| 17 | **ClarificationMiddleware** | `wrap_tool_call` | Y | N | Intercepts `ask_clarification`, returns `Command(goto=END)` |

### 3.3 Hard Ordering Constraints

1. **ThreadDataMiddleware** must precede **SandboxMiddleware** (sandbox needs thread directories)
2. **ClarificationMiddleware** must be **last** in the list so its `after_model` fires **first** (reverse order) to intercept clarification tool calls

---

## 4. Sandbox System

### 4.1 Three Sandbox Modes

| Mode | Provider | Isolation | Requires | Best For |
|------|----------|-----------|----------|----------|
| **Local** | `LocalSandboxProvider` | None (host) | Nothing | Local dev, trusted use |
| **AIO** | `AioSandboxProvider` | Docker container | Docker | Single-user with isolation |
| **Provisioner** | `AioSandboxProvider` + `provisioner_url` | K8s Pod | Docker + K8s | Production, multi-user |

### 4.2 Sandbox Provider Pattern

Abstract `SandboxProvider` with lifecycle methods:

```python
class SandboxProvider(ABC):
    def acquire(thread_id: str) -> str:    # returns sandbox_id
    def get(sandbox_id: str) -> Sandbox:   # returns Sandbox instance
    def release(sandbox_id: str) -> None:  # releases resources
```

Abstract `Sandbox` with execution methods:

```python
class Sandbox(ABC):
    def execute_command(command: str) -> str
    def read_file(path: str) -> str
    def write_file(path: str, content: str) -> None
    def list_dir(path: str) -> list
    def glob(pattern: str) -> list
    def grep(pattern: str, path: str) -> list
    def update_file(path: str, old: str, new: str) -> None
```

Singleton via `get_sandbox_provider()`, resolved from `config.yaml` via reflection.

### 4.3 Local Sandbox

```yaml
sandbox:
  use: deerflow.sandbox.local:LocalSandboxProvider
  allow_host_bash: false  # disabled by default (not a security boundary)
```

- Singleton, direct host filesystem execution
- Bash gated by `allow_host_bash` config
- Path validation via `validate_local_tool_path()` and `validate_local_bash_command_paths()`

### 4.4 AIO Sandbox (Docker)

```yaml
sandbox:
  use: deerflow.community.aio_sandbox:AioSandboxProvider
  # image: enterprise-public-cn-beijing.cr.volces.com/vefaas-public/all-in-one-sandbox:latest
  # port: 8080
  # replicas: 3
  # container_prefix: deer-flow-sandbox
```

Key implementation features:
- Deterministic sandbox IDs from thread IDs (SHA-256)
- Two-layer consistency: in-process cache + backend discovery
- Warm pool for released-but-still-running containers
- Cross-process file locks for concurrent creation
- Idle timeout with background checker thread
- LRU eviction when replicas limit exceeded
- Thread-safe via `threading.Lock` on shell commands
- On macOS: auto-prefers Apple Container, falls back to Docker

### 4.5 Provisioner Sandbox (Kubernetes)

```yaml
sandbox:
  use: deerflow.community.aio_sandbox:AioSandboxProvider
  provisioner_url: http://provisioner:8002
```

The provisioner is a standalone FastAPI service that manages per-sandbox K8s Pods:

```
Backend --> HTTP --> Provisioner(:8002) --> K8s API --> Pod + NodePort Service
Backend --> direct NodePort --> Sandbox Pod(:8080)
```

Pod spec: 100m-1000m CPU, 256Mi-1Gi memory, readiness/liveness probes on `/v1/sandbox:8080`.

### 4.6 Docker-outside-of-Docker (DooD)

When running DeerFlow services in Docker (`make up` / `make docker-start`) with AIO sandbox mode, the `langgraph` container communicates with the **host's Docker daemon** via a mounted socket to create **sibling** sandbox containers:

```
┌─── Host Docker Daemon ─────────────────────────┐
│                                                 │
│  ┌─────────┐  ┌─────────┐  ┌──────────────┐   │
│  │ frontend │  │ gateway │  │  langgraph   │   │
│  └─────────┘  └─────────┘  └──────┬───────┘   │
│                                    │            │
│                          docker.sock│            │
│                                    ▼            │
│  ┌──────────────┐  ┌──────────────┐            │
│  │ aio-sandbox-1│  │ aio-sandbox-2│  (sibling) │
│  └──────────────┘  └──────────────┘            │
│                                                 │
└─────────────────────────────────────────────────┘
```

This is NOT containers-within-containers -- sandbox containers are **siblings** on the host Docker daemon.

**Key**: Running `make up` containerizes the DeerFlow **services**, but does NOT automatically sandbox agent code execution. You still need AIO or provisioner mode for isolated code execution.

### 4.7 Virtual Path Mapping

| Virtual Path | Physical Path |
|---|---|
| `/mnt/user-data/workspace` | `backend/.deer-flow/threads/{thread_id}/user-data/workspace` |
| `/mnt/user-data/uploads` | `backend/.deer-flow/threads/{thread_id}/user-data/uploads` |
| `/mnt/user-data/outputs` | `backend/.deer-flow/threads/{thread_id}/user-data/outputs` |
| `/mnt/skills` | `skills/` |
| `/mnt/acp-workspace/` | `backend/.deer-flow/threads/{thread_id}/acp-workspace/` (read-only) |

### 4.8 Sandbox Tools

Seven LangChain `@tool` functions: `bash`, `ls`, `glob`, `grep`, `read_file`, `write_file`, `str_replace`.

- All use `ensure_sandbox_initialized()` for lazy acquisition
- Output truncation: `bash` middle-truncates at 20KB, `read_file` head-truncates at 50KB
- Host paths are masked in output for security

---

## 5. Model / LLM Provider System

### 5.1 Model Factory

`create_chat_model(name, thinking_enabled, **kwargs)`:
1. Resolves `ModelConfig` from `AppConfig`
2. Resolves the class via `resolve_class()` (e.g., `langchain_openai.ChatOpenAI`)
3. Handles thinking mode: merges `when_thinking_enabled` settings
4. Attaches tracing callbacks

### 5.2 Supported Providers

| `use` Path | Provider |
|------------|----------|
| `langchain_openai:ChatOpenAI` | OpenAI, OpenRouter, Novita, any OpenAI-compatible |
| `langchain_anthropic:ChatAnthropic` | Anthropic Claude |
| `langchain_deepseek:ChatDeepSeek` | DeepSeek |
| `langchain_google_genai:ChatGoogleGenerativeAI` | Google Gemini |
| `deerflow.models.patched_openai:PatchedChatOpenAI` | Gemini with thinking support |
| `deerflow.models.patched_deepseek:PatchedChatDeepSeek` | DeepSeek, Kimi, Doubao |
| `deerflow.models.claude_provider:ClaudeChatModel` | Claude Code OAuth |
| `deerflow.models.openai_codex_provider:CodexChatModel` | Codex CLI |

Special per-model flags: `supports_thinking`, `supports_vision`, `supports_reasoning_effort`, `when_thinking_enabled`, `use_responses_api`, `output_version`.

### 5.3 Claude Provider

`ClaudeChatModel` extends `ChatAnthropic` with:
- OAuth Bearer auth (detects `sk-ant-oat` prefix, loads from multiple credential sources)
- Prompt caching (`cache_control: ephemeral` on system, recent messages, last tool)
- Auto thinking budget (80% of max_tokens)
- OAuth billing header injection
- Retry logic (3 attempts with exponential backoff + Retry-After header)

---

## 6. Tool System

### 6.1 Tool Assembly

`get_available_tools()` assembles tools from four sources:

1. **Config-defined tools** -- resolved from `config.yaml` via reflection (web_search, web_fetch, image_search, file tools, bash)
2. **MCP tools** -- from enabled MCP servers (lazy initialized, cached with mtime invalidation)
3. **Built-in tools** -- `present_files`, `ask_clarification`, `view_image` (vision models only)
4. **Subagent tool** (if enabled) -- `task` for delegating to subagents

### 6.2 Tool Search (Deferred Loading)

When `tool_search.enabled: true` in config, MCP tools are not loaded into the model context directly. Instead they are listed by name and discoverable via a `tool_search` tool at runtime, reducing context token usage.

### 6.3 Community Tool Integrations

| Integration | Tools | API Key Required |
|-------------|-------|:---:|
| **Tavily** | `web_search`, `web_fetch` | Yes |
| **DuckDuckGo** | `web_search` | No |
| **Jina AI** | `web_fetch` | Yes |
| **Firecrawl** | Web scraping | Yes |
| **InfoQuest** | Search & crawl (BytePlus) | Yes |
| **Image Search** | DuckDuckGo image search | No |

---

## 7. Skills System

### 7.1 Skill Structure

Skills are directories under `skills/{public,custom}/` containing a `SKILL.md` with YAML frontmatter:

```markdown
---
name: skill-name
description: What the skill does
dependency: optional-dependency
allowed-tools:
  - bash
  - write_file
---

## Instructions for the agent...
```

Optional subdirectories: `scripts/`, `references/`, `templates/`, `agents/`.

### 7.2 Built-in Skills

19 public skills including:
- **deep-research** -- Systematic multi-phase web research (broad -> deep dive -> validation -> synthesis)
- **chart-visualization** -- 26 chart types via AntV with Node.js generation script
- **data-analysis** -- Python-based data analysis with pandas
- **bootstrap** -- Conversational onboarding to generate `SOUL.md` personality
- **skill-creator** -- Meta-skill for creating new skills
- **frontend-design** -- Web design
- **image/video/podcast/ppt-generation** -- Content generation skills
- **github-deep-research** -- GitHub API-based research
- **claude-to-deerflow** -- Claude Code bridge
- **find-skills** -- Skill discovery and installation
- **vercel-deploy-claimable** -- Vercel deployment automation

### 7.3 Progressive Loading

Skills are not all loaded into context at once. They are progressively injected only when the task requires them, reducing token usage.

---

## 8. Sub-Agent System

The lead agent can spawn sub-agents for complex tasks via the `task` tool.

**Built-in subagent types:**
- `general-purpose` -- All tools except `task` (prevents recursive delegation)
- `bash` -- Command execution specialist

**Execution model:**
- Dual thread pools: 3 scheduler workers + 3 execution workers
- `MAX_CONCURRENT_SUBAGENTS = 3`, enforced by `SubagentLimitMiddleware`
- 15-minute timeout (configurable per-agent)
- Sub-agents run in isolated context with their own middleware chain

**ACP Agents:** External ACP-compatible agents (Claude Code, Codex) can be invoked via `invoke_acp_agent` tool with per-thread workspace isolation.

---

## 9. Memory System

### 9.1 Architecture

Four-layer pipeline: **Middleware -> Queue -> Updater -> Storage**

```
MemoryMiddleware          MemoryUpdateQueue       MemoryUpdater        FileMemoryStorage
  (after_agent)    --->    (debounce 30s)   --->   (LLM call)   --->   (JSON file)
                                                                            │
                                                                            ▼
                                                                  Next request injects
                                                                  via system prompt
```

### 9.2 Data Structure

Stored in `backend/.deer-flow/memory.json`:

```json
{
  "userContext": {
    "workContext": "...",
    "personalContext": "...",
    "topOfMind": "..."
  },
  "history": {
    "recentMonths": "...",
    "earlierContext": "...",
    "longTermBackground": "..."
  },
  "facts": [
    {
      "id": "uuid",
      "content": "...",
      "category": "...",
      "confidence": 0.85,
      "createdAt": "2026-01-15T10:00:00Z",
      "source": "conversation"
    }
  ]
}
```

### 9.3 Update Pipeline

1. `MemoryMiddleware.after_agent` filters messages (user + final AI, strips `<uploaded_files>`)
2. `MemoryUpdateQueue.add()` queues per-thread, resets debounce timer
3. After 30s idle, `MemoryUpdater.update_memory()` fires:
   - Gets current memory from storage
   - Calls LLM with structured `MEMORY_UPDATE_PROMPT`
   - Parses JSON response, applies updates with deduplication
   - Enforces `max_facts` (100) limit by confidence ranking
4. `FileMemoryStorage.save()` -- atomic writes via temp-file + rename
5. Next request: `format_memory_for_injection()` injects top 15 facts (token-budgeted) into system prompt

**Configuration:**

```yaml
memory:
  enabled: true
  injection_enabled: true
  storage_path: null  # defaults to backend/.deer-flow/memory.json
  debounce_seconds: 30
  model_name: null  # defaults to global default
  max_facts: 100
  fact_confidence_threshold: 0.7
  max_injection_tokens: 2000
```

---

## 10. MCP Integration

Uses `langchain-mcp-adapters` `MultiServerMCPClient`. Supports three transports: **stdio**, **SSE**, and **HTTP**.

Key features:
- OAuth token flows (`client_credentials`, `refresh_token`) for HTTP/SSE servers
- Configuration in `extensions_config.json`
- Lazy initialization with file mtime cache invalidation
- Runtime updates via Gateway API
- Tool interceptors for OAuth header injection
- Async-only tools patched with sync wrappers (10-worker thread pool)

---

## 11. Guardrails System

Pre-tool-call authorization middleware with three provider options:

| Provider | Description |
|----------|-------------|
| **AllowlistProvider** | Built-in, zero deps -- block/allow tools by name |
| **OAP Passport Provider** | Policy-based using Open Agent Passport standard |
| **Custom Provider** | Any class with `evaluate`/`aevaluate` methods |

Default `fail_closed: true` -- if provider errors, tool call is blocked. `GraphBubbleUp` (LangGraph control signals) always propagated.

---

## 12. Gateway API

FastAPI REST API with 13 routers:

| Router | Path | Purpose |
|--------|------|---------|
| Models | `/api/models` | List/detail LLM models |
| MCP | `/api/mcp/config` | Get/update MCP server config |
| Skills | `/api/skills` | List/detail/enable/disable/install skills |
| Memory | `/api/memory` | Get/reload memory, config, status, facts CRUD |
| Uploads | `/api/threads/{id}/uploads` | Upload/list/delete files (auto PDF/PPT/Excel/Word -> MD) |
| Threads | `/api/threads/{id}` | Delete local thread data |
| Artifacts | `/api/threads/{id}/artifacts` | Serve generated files (XSS-safe) |
| Suggestions | `/api/threads/{id}/suggestions` | Generate follow-up questions |
| Agents | `/api/agents` | Custom agent CRUD |
| Channels | `/api/channels` | IM channel management |
| Thread Runs | -- | LangGraph-compatible runs lifecycle |
| Runs | -- | Stateless runs (stream/wait) |
| Assistants Compat | -- | LangGraph Platform stub |

---

## 13. Frontend Architecture

### 13.1 Tech Stack

- **Next.js 16** with App Router, **React 19**, **TypeScript 5.8**
- **Tailwind CSS 4** (new `@import` syntax)
- **pnpm 10.26.2** package manager
- **@langchain/langgraph-sdk** -- primary client for LangGraph backend
- **@tanstack/react-query** -- server state management
- **Vercel AI SDK v6** -- AI-specific UI elements
- **Radix UI** -- headless primitives
- **CodeMirror** -- code editor for artifacts
- **@xyflow/react** -- flow/node graph visualization
- **streamdown** -- streaming markdown renderer
- **rehype-katex / remark-math** -- LaTeX math
- **shiki** -- syntax highlighting
- **nextra** -- documentation pages
- **better-auth** -- authentication (not yet active)

### 13.2 App Router Structure

| Route | Purpose |
|-------|---------|
| `/` | Landing page |
| `/workspace/` | Workspace root (sidebar, command palette) |
| `/workspace/chats/[thread_id]` | Main chat page |
| `/workspace/agents/[agent_name]/chats/[thread_id]` | Agent-specific chat |
| `/workspace/agents/new` | New agent creation |
| `/[lang]/docs/` | Nextra-powered documentation |

### 13.3 Core Business Logic

Located in `frontend/src/core/`:

- **`api/api-client.ts`** -- Singleton `LangGraphClient`, URL from env or defaults to `{origin}/api/langgraph`
- **`threads/`** -- `useThreadStream()` (central streaming hook), `useThreads()` (paginated search), `useDeleteThread()`, `useRenameThread()`
- **`settings/local.ts`** -- localStorage persistence for model, mode, layout preferences
- **`agents/`** -- Agent CRUD
- **`artifacts/`** -- Artifact loading and caching
- **`i18n/`** -- en-US and zh-CN locale support
- **`memory/`** -- Memory API hooks
- **`mcp/`** -- MCP configuration hooks
- **`models/`** -- Model listing hooks
- **`skills/`** -- Skills management hooks
- **`uploads/`** -- File upload validation and hooks

### 13.4 Data Flow Pattern

```
User types message in InputBox
    --> ChatPage.handleSubmit()
    --> uploadFiles() to Gateway /api/threads/{id}/uploads
    --> thread.submit() to LangGraph SDK (streaming)
    --> Stream events: onCreated, onLangChainEvent, onUpdateEvent, onCustomEvent
    --> ThreadState updated reactively
    --> TanStack Query caches thread lists
```

### 13.5 Chat Modes

| Mode | Thinking | Plan Mode | Sub-agents |
|------|:--------:|:---------:|:----------:|
| **Flash** | No | No | No |
| **Thinking** | Yes | No | No |
| **Pro** | Yes | Yes | No |
| **Ultra** | Yes | Yes | Yes |

---

## 14. Configuration System

Two config files at the project root:

1. **`config.yaml`** -- main application config (version 5)
2. **`extensions_config.json`** -- MCP servers and skill enabled/disabled state

**Config resolution priority:** explicit path > `$DEER_FLOW_CONFIG_PATH` env var > `config.yaml` in CWD > `config.yaml` in parent directory.

**Key features:**
- Values starting with `$` resolved as environment variables
- Cached singleton with automatic mtime-based reload
- Runtime overrides via `ContextVar` (`push_current_app_config` / `pop_current_app_config`)
- Schema versioning with `make config-upgrade` for auto-merging new fields

**`AppConfig`** aggregates: `models`, `sandbox`, `tools`, `tool_groups`, `skills`, `extensions`, `tool_search`, `title`, `summarization`, `memory`, `subagents`, `guardrails`, `checkpointer`, `stream_bridge`, `token_usage`.

---

## 15. Docker Infrastructure & Deployment

### 15.1 Development Commands

| Command | Description |
|---------|-------------|
| `make config` | Generate `config.yaml` from example |
| `make install` | Install backend (uv sync) + frontend (pnpm install) dependencies |
| `make dev` | Start all services locally with hot-reload |
| `make start` | Start all services locally in production mode |
| `make docker-init` | Pre-pull sandbox container image |
| `make docker-start` | Build & start Docker dev environment |
| `make docker-stop` | Stop Docker services |
| `make up` | Production Docker deployment |
| `make down` | Stop production containers |
| `make setup-sandbox` | Pre-pull AIO sandbox image |

### 15.2 Production Compose

`docker/docker-compose.yaml` runs 4-5 services on a bridge network:

- **nginx** -- alpine image, port 2026 exposed
- **frontend** -- multi-stage Dockerfile `prod` target, optimized build
- **gateway** -- Python 3.12, 2 uvicorn workers, no reload
- **langgraph** -- Python 3.12, `langgraph dev --no-browser --no-reload`
- **provisioner** (optional, `--profile provisioner`) -- K8s pod manager

Both gateway and langgraph mount `/var/run/docker.sock` for DooD sandbox containers, plus `~/.claude` and `~/.codex` read-only for CLI auth forwarding.

### 15.3 Development Compose

`docker/docker-compose-dev.yaml` -- same services with:
- Frontend mounts `src/`, `public/` for hot-reload with `WATCHPACK_POLLING=true`
- Gateway/langgraph mount entire `backend/` with named volumes for `.venv` preservation
- `--reload` enabled on gateway; `uv sync` runs on start
- Subnet: `192.168.200.0/24`

### 15.4 Nginx Routing

```
/api/langgraph/*    --> rewrite to /* --> langgraph:2024 (SSE: buffering off, 600s timeout)
/api/models         --> gateway:8001
/api/memory         --> gateway:8001
/api/mcp            --> gateway:8001
/api/skills         --> gateway:8001
/api/agents         --> gateway:8001
/api/threads/*/uploads --> gateway:8001 (client_max_body_size 100M)
/api/threads/*      --> gateway:8001
/api/sandboxes      --> provisioner:8002 (optional)
/docs, /redoc       --> gateway:8001
/*                  --> frontend:3000 (WebSocket upgrade supported)
```

---

## 16. Runtime System

### 16.1 Run Manager

`RunManager` -- in-memory run registry with async lock protection:
- `create()` / `create_or_reject()` -- atomic inflight check + create
- `cancel()` with interrupt/rollback actions
- Multitask strategies: `reject`, `interrupt`, `rollback`
- `cleanup()` with delayed record removal

### 16.2 Stream Bridge

Decouples agent workers (producers) from SSE endpoints (consumers):

```python
class StreamBridge(ABC):
    def publish(event: StreamEvent) -> None
    def publish_end() -> None
    def subscribe() -> AsyncIterator[StreamEvent]  # with heartbeat
    def cleanup() -> None
```

**MemoryStreamBridge** implementation:
- Per-run `asyncio.Queue` (maxsize 256)
- 30-second publish timeout
- Critical END_SENTINEL delivery: evicts oldest events if queue full
- 15-second heartbeat interval
- Redis backend planned for Phase 2

---

## 17. IM Channels

Supports multiple messaging platforms -- no public IP required:

| Platform | Protocol |
|----------|----------|
| **Telegram** | Long-polling |
| **Slack** | Socket Mode |
| **Feishu/Lark** | WebSocket |
| **WeCom** | WebSocket |

Channels communicate with LangGraph Server via `langgraph-sdk` HTTP client. Commands: `/new`, `/status`, `/models`, `/memory`, `/help`.

---

## 18. Design Patterns & Principles

### Reflection-Based Resolution
Models, tools, sandbox providers, and guardrail providers are all loaded via `resolve_class()` / `resolve_variable()` -- parsing `"module.path:ClassName"` strings from config. This enables complete extensibility without code changes.

### Provider Pattern
Abstract providers with acquire/get/release lifecycle: `SandboxProvider`, `GuardrailProvider`, `CheckpointerProvider`, `StreamBridgeProvider`. Implementations are swappable via config.

### Middleware Pipeline
Not an onion model -- a linear pipeline with hook-based interception points. Forward execution for `before_*`, reverse for `after_*`. Enables orthogonal cross-cutting concerns.

### Progressive Loading
Skills, MCP tools (via tool search), and memory facts are loaded incrementally to minimize context token usage.

### Strict Dependency Boundaries
Harness (framework) never imports App (application). Enforced by CI tests. Enables the harness to be published as an independent package.

### Config-Driven Architecture
Nearly everything is configurable via `config.yaml` without code changes: models, tools, sandbox, skills, middleware features, guardrails, memory, summarization, subagents.

### Thread Isolation
Each conversation thread gets isolated directories (workspace, uploads, outputs), sandbox instances, and state. No cross-thread leakage.

### Atomic Operations
Memory storage uses temp-file + rename for crash-safe writes. Config auto-reloads on mtime change. Sandbox creation uses cross-process file locks.

### Embedded Client
`DeerFlowClient` provides direct in-process access without HTTP services, enabling library-style usage of the harness.

### Docker-outside-of-Docker (DooD)
Rather than Docker-in-Docker (DinD), the system mounts the host Docker socket to create sibling containers. This avoids nested container complexity while providing isolation.
