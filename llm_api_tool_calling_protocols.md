# LLM APIs: Comprehensive Guide to OpenAI, Claude, and Gemini

A comprehensive guide to LLM APIs from major providers, covering models, authentication, endpoints, tool calling, vision, embeddings, streaming, and more.

---

## Table of Contents

1. [Overview](#overview)
2. [OpenAI API](#openai-api)
   - [Authentication](#openai-authentication)
   - [Models](#openai-models)
   - [Chat Completions](#openai-chat-completions)
   - [Tool Calling](#openai-tool-calling)
   - [Vision](#openai-vision)
   - [Embeddings](#openai-embeddings)
   - [Streaming](#openai-streaming)
   - [Structured Outputs](#openai-structured-outputs)
3. [Anthropic Claude API](#anthropic-claude-api)
   - [Authentication](#claude-authentication)
   - [Models](#claude-models)
   - [Messages API](#claude-messages-api)
   - [Tool Calling](#claude-tool-calling)
   - [Vision](#claude-vision)
   - [Extended Thinking](#claude-extended-thinking)
   - [Streaming](#claude-streaming)
   - [Prompt Caching](#claude-prompt-caching)
4. [Google Gemini API](#google-gemini-api)
   - [Authentication](#gemini-authentication)
   - [Models](#gemini-models)
   - [Generate Content](#gemini-generate-content)
   - [Tool Calling](#gemini-tool-calling)
   - [Vision & Multimodal](#gemini-vision)
   - [Embeddings](#gemini-embeddings)
   - [Streaming](#gemini-streaming)
   - [Context Caching](#gemini-context-caching)
5. [Comparison Tables](#comparison-tables)
6. [Agent Loop Patterns](#agent-loop-patterns)
7. [Best Practices](#best-practices)
8. [Rate Limits & Pricing](#rate-limits--pricing)
9. [References](#references)

---

## Overview

Modern LLM APIs provide capabilities for:
- **Chat Completions**: Conversational AI with system prompts
- **Tool/Function Calling**: Structured interaction with external tools
- **Vision**: Image understanding and analysis
- **Embeddings**: Vector representations for semantic search
- **Streaming**: Real-time token-by-token responses
- **Structured Outputs**: JSON schema-constrained responses

### Key Concepts

| Term | Description |
|------|-------------|
| **Token** | Basic unit of text processing (~4 chars in English) |
| **Context Window** | Maximum tokens (input + output) in a conversation |
| **System Prompt** | Instructions that set AI behavior |
| **Temperature** | Controls randomness (0 = deterministic, 1+ = creative) |
| **Tool/Function** | External capability the LLM can invoke |
| **Embedding** | Dense vector representation of text |

---

## OpenAI API

### OpenAI Authentication

**API Key Format**: `sk-...` (starts with `sk-`)

```bash
# Environment variable
export OPENAI_API_KEY="sk-..."

# Header
Authorization: Bearer sk-...
```

**Python SDK Installation**:
```bash
pip install openai
```

**Python SDK Setup**:
```python
from openai import OpenAI

# Auto-reads OPENAI_API_KEY env var
client = OpenAI()

# Or explicit
client = OpenAI(api_key="sk-...")

# Azure OpenAI
from openai import AzureOpenAI
client = AzureOpenAI(
    api_key="...",
    api_version="2024-02-15-preview",
    azure_endpoint="https://YOUR_RESOURCE.openai.azure.com"
)
```

### OpenAI Models

| Model | Context | Input $/1M | Output $/1M | Description |
|-------|---------|------------|-------------|-------------|
| **gpt-4o** | 128K | $2.50 | $10.00 | Flagship multimodal, fast |
| **gpt-4o-mini** | 128K | $0.15 | $0.60 | Affordable, capable |
| **gpt-4-turbo** | 128K | $10.00 | $30.00 | Previous flagship |
| **gpt-4** | 8K | $30.00 | $60.00 | Original GPT-4 |
| **gpt-3.5-turbo** | 16K | $0.50 | $1.50 | Legacy, fast |
| **o1** | 200K | $15.00 | $60.00 | Reasoning model |
| **o1-mini** | 128K | $3.00 | $12.00 | Faster reasoning |
| **o3-mini** | 200K | $1.10 | $4.40 | Latest reasoning |

**Embeddings Models**:
| Model | Dimensions | $/1M tokens |
|-------|------------|-------------|
| text-embedding-3-large | 3072 | $0.13 |
| text-embedding-3-small | 1536 | $0.02 |
| text-embedding-ada-002 | 1536 | $0.10 |

### OpenAI Chat Completions

**Endpoint**:
```
POST https://api.openai.com/v1/chat/completions
```

**Basic Request**:
```json
{
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7,
    "max_tokens": 1000,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0
}
```

**Response**:
```json
{
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1699000000,
    "model": "gpt-4o",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "Hello! How can I help you today?"
        },
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "total_tokens": 30
    }
}
```

**Python Example**:
```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
```

### OpenAI Tool Calling

**Request with Tools**:
```json
{
    "model": "gpt-4o",
    "messages": [
        {"role": "user", "content": "What's the weather in Tokyo?"}
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name, e.g., 'Tokyo, Japan'"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"]
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ],
    "tool_choice": "auto"
}
```

**Tool Choice Options**:
| Value | Behavior |
|-------|----------|
| `"auto"` | LLM decides (default) |
| `"none"` | Never call tools |
| `"required"` | Must call at least one tool |
| `{"type": "function", "function": {"name": "..."}}` | Force specific tool |

**Response with Tool Call**:
```json
{
    "choices": [{
        "message": {
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{\"location\": \"Tokyo, Japan\", \"unit\": \"celsius\"}"
                }
            }]
        },
        "finish_reason": "tool_calls"
    }]
}
```

**Sending Tool Results**:
```json
{
    "model": "gpt-4o",
    "messages": [
        {"role": "user", "content": "What's the weather in Tokyo?"},
        {
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{\"location\": \"Tokyo, Japan\"}"
                }
            }]
        },
        {
            "role": "tool",
            "tool_call_id": "call_abc123",
            "content": "{\"temperature\": 22, \"condition\": \"sunny\"}"
        }
    ]
}
```

**Parallel Tool Calls**:
```python
# OpenAI can return multiple tool calls in one response
response.choices[0].message.tool_calls  # List of tool calls
# [call_1, call_2, call_3] - execute all, return all results
```

### OpenAI Vision

**Request with Image URL**:
```json
{
    "model": "gpt-4o",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/image.jpg",
                    "detail": "high"
                }
            }
        ]
    }],
    "max_tokens": 300
}
```

**Request with Base64 Image**:
```python
import base64

with open("image.jpg", "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_data}"
                }
            }
        ]
    }]
)
```

**Detail Levels**:
| Detail | Tokens | Use Case |
|--------|--------|----------|
| `low` | 85 fixed | Quick classification |
| `high` | 85-1105 | Detailed analysis |
| `auto` | Varies | Let API decide |

### OpenAI Embeddings

**Endpoint**:
```
POST https://api.openai.com/v1/embeddings
```

**Request**:
```json
{
    "model": "text-embedding-3-large",
    "input": "The quick brown fox jumps over the lazy dog",
    "dimensions": 1536
}
```

**Response**:
```json
{
    "object": "list",
    "data": [{
        "object": "embedding",
        "index": 0,
        "embedding": [0.0023064255, -0.009327292, ...]
    }],
    "model": "text-embedding-3-large",
    "usage": {
        "prompt_tokens": 9,
        "total_tokens": 9
    }
}
```

**Python Example**:
```python
response = client.embeddings.create(
    model="text-embedding-3-large",
    input=["Hello world", "Goodbye world"],
    dimensions=1536  # Optional: reduce dimensions
)

embeddings = [d.embedding for d in response.data]
```

### OpenAI Streaming

**Request**:
```json
{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
}
```

**Python Streaming**:
```python
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

**Async Streaming**:
```python
from openai import AsyncOpenAI

async_client = AsyncOpenAI()

async def stream_response():
    stream = await async_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True
    )

    async for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")
```

### OpenAI Structured Outputs

**JSON Mode**:
```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "List 3 colors as JSON"}],
    response_format={"type": "json_object"}
)
```

**Structured Outputs with Schema**:
```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    city: str

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Extract: John is 30 from NYC"}],
    response_format=Person
)

person = response.choices[0].message.parsed
print(person.name, person.age)  # John 30
```

---

## Anthropic Claude API

### Claude Authentication

**API Key Format**: `sk-ant-...` (starts with `sk-ant-`)

```bash
# Environment variable
export ANTHROPIC_API_KEY="sk-ant-..."

# Headers
x-api-key: sk-ant-...
anthropic-version: 2023-06-01
```

**Python SDK Installation**:
```bash
pip install anthropic
```

**Python SDK Setup**:
```python
import anthropic

# Auto-reads ANTHROPIC_API_KEY env var
client = anthropic.Anthropic()

# Or explicit
client = anthropic.Anthropic(api_key="sk-ant-...")

# Async client
async_client = anthropic.AsyncAnthropic()
```

### Claude Models

| Model | Context | Input $/1M | Output $/1M | Description |
|-------|---------|------------|-------------|-------------|
| **claude-opus-4-20250514** | 200K | $15.00 | $75.00 | Most capable, complex tasks |
| **claude-sonnet-4-20250514** | 200K | $3.00 | $15.00 | Balanced performance/cost |
| **claude-3-5-haiku-20241022** | 200K | $0.80 | $4.00 | Fast, efficient |
| **claude-3-opus-20240229** | 200K | $15.00 | $75.00 | Previous flagship |
| **claude-3-sonnet-20240229** | 200K | $3.00 | $15.00 | Previous balanced |
| **claude-3-haiku-20240307** | 200K | $0.25 | $1.25 | Previous fast |

**Model Aliases** (always point to latest):
- `claude-opus-4-5-20250929` - Latest Opus
- `claude-sonnet-4-20250514` - Latest Sonnet
- `claude-3-5-haiku-latest` - Latest Haiku

### Claude Messages API

**Endpoint**:
```
POST https://api.anthropic.com/v1/messages
```

**Basic Request**:
```json
{
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 1024,
    "system": "You are a helpful assistant.",
    "messages": [
        {"role": "user", "content": "Hello!"}
    ]
}
```

**Response**:
```json
{
    "id": "msg_abc123",
    "type": "message",
    "role": "assistant",
    "content": [{
        "type": "text",
        "text": "Hello! How can I help you today?"
    }],
    "model": "claude-sonnet-4-20250514",
    "stop_reason": "end_turn",
    "usage": {
        "input_tokens": 15,
        "output_tokens": 10
    }
}
```

**Python Example**:
```python
import anthropic

client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system="You are a helpful assistant.",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

print(message.content[0].text)
```

**Multi-turn Conversation**:
```python
messages = [
    {"role": "user", "content": "My name is Alice."},
    {"role": "assistant", "content": "Nice to meet you, Alice!"},
    {"role": "user", "content": "What's my name?"}
]

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=100,
    messages=messages
)
```

### Claude Tool Calling

**Request with Tools**:
```json
{
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 1024,
    "tools": [
        {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }
    ],
    "messages": [
        {"role": "user", "content": "What's the weather in Tokyo?"}
    ]
}
```

**Tool Choice Options**:
```json
{"type": "auto"}                          // LLM decides (default)
{"type": "any"}                           // Must use at least one tool
{"type": "tool", "name": "get_weather"}   // Force specific tool
```

**Response with Tool Use**:
```json
{
    "id": "msg_abc123",
    "content": [
        {
            "type": "tool_use",
            "id": "toolu_abc123",
            "name": "get_weather",
            "input": {
                "location": "Tokyo, Japan",
                "unit": "celsius"
            }
        }
    ],
    "stop_reason": "tool_use"
}
```

**Sending Tool Results**:
```json
{
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 1024,
    "messages": [
        {"role": "user", "content": "What's the weather in Tokyo?"},
        {
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": "toolu_abc123",
                "name": "get_weather",
                "input": {"location": "Tokyo, Japan"}
            }]
        },
        {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_abc123",
                "content": "{\"temperature\": 22, \"condition\": \"sunny\"}"
            }]
        }
    ]
}
```

**Tool Result with Error**:
```json
{
    "type": "tool_result",
    "tool_use_id": "toolu_abc123",
    "is_error": true,
    "content": "Location not found"
}
```

### Claude Vision

**Request with Image**:
```json
{
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 1024,
    "messages": [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": "/9j/4AAQSkZJRg..."
                }
            },
            {
                "type": "text",
                "text": "What's in this image?"
            }
        ]
    }]
}
```

**Image via URL**:
```json
{
    "type": "image",
    "source": {
        "type": "url",
        "url": "https://example.com/image.jpg"
    }
}
```

**Python Vision Example**:
```python
import base64

with open("image.jpg", "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_data
                }
            },
            {"type": "text", "text": "Describe this image in detail."}
        ]
    }]
)
```

**Supported Image Types**:
- image/jpeg
- image/png
- image/gif
- image/webp

### Claude Extended Thinking

Claude supports extended thinking for complex reasoning tasks:

**Request**:
```json
{
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 16000,
    "thinking": {
        "type": "enabled",
        "budget_tokens": 10000
    },
    "messages": [
        {"role": "user", "content": "Solve this complex math problem..."}
    ]
}
```

**Response with Thinking**:
```json
{
    "content": [
        {
            "type": "thinking",
            "thinking": "Let me break this down step by step...\n\nFirst, I need to..."
        },
        {
            "type": "text",
            "text": "The answer is 42. Here's my reasoning..."
        }
    ]
}
```

**Python Example**:
```python
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000
    },
    messages=[{"role": "user", "content": "Solve: ..."}]
)

for block in response.content:
    if block.type == "thinking":
        print("Thinking:", block.thinking)
    elif block.type == "text":
        print("Answer:", block.text)
```

### Claude Streaming

**Python Streaming**:
```python
with client.messages.stream(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Tell me a story"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

**Async Streaming**:
```python
async with async_client.messages.stream(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}]
) as stream:
    async for text in stream.text_stream:
        print(text, end="")
```

**Raw SSE Events**:
```python
with client.messages.stream(...) as stream:
    for event in stream:
        if event.type == "content_block_delta":
            print(event.delta.text, end="")
        elif event.type == "message_stop":
            print("\n[Done]")
```

### Claude Prompt Caching

Cache long system prompts or documents to reduce costs:

**Request with Cache**:
```json
{
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 1024,
    "system": [
        {
            "type": "text",
            "text": "You are an expert on this 100-page document...",
            "cache_control": {"type": "ephemeral"}
        }
    ],
    "messages": [
        {"role": "user", "content": "Summarize chapter 3"}
    ]
}
```

**Pricing**:
- Cache write: 25% more than base input price
- Cache read: 90% less than base input price
- Cache TTL: 5 minutes (extended on hit)

---

## Google Gemini API

### Gemini Authentication

**API Key**: Get from [Google AI Studio](https://aistudio.google.com/app/apikey)

```bash
# Environment variable
export GOOGLE_API_KEY="..."

# URL parameter (AI Studio)
?key=YOUR_API_KEY

# Header (Vertex AI)
Authorization: Bearer $(gcloud auth print-access-token)
```

**Python SDK Installation**:
```bash
pip install google-generativeai
```

**Python SDK Setup**:
```python
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")

# Or with environment variable (auto-configured)
# export GOOGLE_API_KEY="..."
```

**Vertex AI Setup**:
```python
import vertexai
from vertexai.generative_models import GenerativeModel

vertexai.init(project="PROJECT_ID", location="us-central1")
model = GenerativeModel("gemini-1.5-pro")
```

### Gemini Models

| Model | Context | Input $/1M | Output $/1M | Description |
|-------|---------|------------|-------------|-------------|
| **gemini-2.0-flash** | 1M | $0.10 | $0.40 | Latest, fastest |
| **gemini-2.0-flash-thinking** | 1M | $0.10 | $0.40 | With reasoning |
| **gemini-1.5-pro** | 2M | $1.25 | $5.00 | Most capable |
| **gemini-1.5-flash** | 1M | $0.075 | $0.30 | Fast, efficient |
| **gemini-1.5-flash-8b** | 1M | $0.0375 | $0.15 | Smallest, fastest |
| **gemini-1.0-pro** | 32K | $0.50 | $1.50 | Legacy |

**Free Tier** (AI Studio):
- 15 RPM (requests per minute)
- 1M TPM (tokens per minute)
- 1500 RPD (requests per day)

**Embeddings Models**:
| Model | Dimensions | Description |
|-------|------------|-------------|
| text-embedding-004 | 768 | Latest, best quality |
| embedding-001 | 768 | Legacy |

### Gemini Generate Content

**Endpoint** (AI Studio):
```
POST https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key=API_KEY
```

**Endpoint** (Vertex AI):
```
POST https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/{REGION}/publishers/google/models/gemini-1.5-pro:generateContent
```

**Basic Request**:
```json
{
    "contents": [{
        "role": "user",
        "parts": [{"text": "Hello!"}]
    }],
    "systemInstruction": {
        "parts": [{"text": "You are a helpful assistant."}]
    },
    "generationConfig": {
        "temperature": 0.7,
        "maxOutputTokens": 1000,
        "topP": 0.95,
        "topK": 40
    }
}
```

**Response**:
```json
{
    "candidates": [{
        "content": {
            "role": "model",
            "parts": [{"text": "Hello! How can I help you today?"}]
        },
        "finishReason": "STOP"
    }],
    "usageMetadata": {
        "promptTokenCount": 10,
        "candidatesTokenCount": 8,
        "totalTokenCount": 18
    }
}
```

**Python Example**:
```python
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    system_instruction="You are a helpful assistant."
)

response = model.generate_content("What is the capital of France?")
print(response.text)
```

**Chat Session**:
```python
model = genai.GenerativeModel("gemini-1.5-pro")
chat = model.start_chat(history=[])

response = chat.send_message("My name is Alice.")
print(response.text)

response = chat.send_message("What's my name?")
print(response.text)
```

### Gemini Tool Calling

**Request with Tools**:
```json
{
    "contents": [{
        "role": "user",
        "parts": [{"text": "What's the weather in Tokyo?"}]
    }],
    "tools": [{
        "function_declarations": [{
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }]
    }],
    "tool_config": {
        "function_calling_config": {
            "mode": "AUTO"
        }
    }
}
```

**Function Calling Modes**:
| Mode | Behavior |
|------|----------|
| `AUTO` | Model decides (default) |
| `ANY` | Must call a function |
| `NONE` | Never call functions |

**Response with Function Call**:
```json
{
    "candidates": [{
        "content": {
            "role": "model",
            "parts": [{
                "functionCall": {
                    "name": "get_weather",
                    "args": {
                        "location": "Tokyo, Japan",
                        "unit": "celsius"
                    }
                }
            }]
        }
    }]
}
```

**Sending Function Results**:
```json
{
    "contents": [
        {"role": "user", "parts": [{"text": "What's the weather in Tokyo?"}]},
        {"role": "model", "parts": [{"functionCall": {"name": "get_weather", "args": {"location": "Tokyo"}}}]},
        {"role": "user", "parts": [{"functionResponse": {"name": "get_weather", "response": {"temperature": 22, "condition": "sunny"}}}]}
    ]
}
```

**Python with Automatic Function Execution**:
```python
def get_weather(location: str) -> dict:
    """Get weather for a location."""
    return {"temperature": 22, "condition": "sunny"}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    tools=[get_weather]
)

response = model.generate_content("What's the weather in Tokyo?")
```

### Gemini Vision

**Request with Image**:
```json
{
    "contents": [{
        "role": "user",
        "parts": [
            {"text": "What's in this image?"},
            {
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": "base64_encoded_image_data"
                }
            }
        ]
    }]
}
```

**Python Vision Example**:
```python
import PIL.Image

model = genai.GenerativeModel("gemini-1.5-pro")

image = PIL.Image.open("image.jpg")
response = model.generate_content(["What's in this image?", image])
print(response.text)
```

**Video Support**:
```python
# Upload video file
video_file = genai.upload_file("video.mp4")

# Wait for processing
while video_file.state.name == "PROCESSING":
    time.sleep(10)
    video_file = genai.get_file(video_file.name)

# Generate content
response = model.generate_content([video_file, "Summarize this video"])
```

**Supported Media Types**:
- Images: JPEG, PNG, GIF, WebP
- Video: MP4, MPEG, MOV, AVI, FLV, MKV, WebM
- Audio: WAV, MP3, AIFF, AAC, OGG, FLAC
- Documents: PDF, plain text

### Gemini Embeddings

**Request**:
```json
{
    "model": "models/text-embedding-004",
    "content": {
        "parts": [{"text": "The quick brown fox"}]
    },
    "taskType": "RETRIEVAL_DOCUMENT",
    "title": "Document title"
}
```

**Task Types**:
| Type | Use Case |
|------|----------|
| `RETRIEVAL_QUERY` | Search queries |
| `RETRIEVAL_DOCUMENT` | Documents to be searched |
| `SEMANTIC_SIMILARITY` | Similarity comparison |
| `CLASSIFICATION` | Text classification |
| `CLUSTERING` | Grouping similar texts |

**Python Example**:
```python
result = genai.embed_content(
    model="models/text-embedding-004",
    content="Hello world",
    task_type="RETRIEVAL_DOCUMENT"
)

embedding = result['embedding']  # 768-dim vector
```

**Batch Embeddings**:
```python
result = genai.embed_content(
    model="models/text-embedding-004",
    content=["Text 1", "Text 2", "Text 3"],
    task_type="RETRIEVAL_DOCUMENT"
)

embeddings = result['embedding']  # List of embeddings
```

### Gemini Streaming

**Python Streaming**:
```python
model = genai.GenerativeModel("gemini-1.5-pro")

response = model.generate_content(
    "Tell me a story",
    stream=True
)

for chunk in response:
    print(chunk.text, end="", flush=True)
```

**Async Streaming**:
```python
async for chunk in await model.generate_content_async(
    "Tell me a story",
    stream=True
):
    print(chunk.text, end="")
```

### Gemini Context Caching

Cache large documents for repeated queries:

```python
# Create cache
cache = genai.caching.CachedContent.create(
    model="gemini-1.5-pro",
    display_name="my-cache",
    contents=[document],
    ttl=datetime.timedelta(hours=1)
)

# Use cached content
model = genai.GenerativeModel.from_cached_content(cache)
response = model.generate_content("Summarize the document")

# Delete cache
cache.delete()
```

**Code Execution**:
Gemini can execute Python code:
```python
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    tools="code_execution"
)

response = model.generate_content("Calculate the first 10 fibonacci numbers")
# Model will write and execute Python code
```

---

## Comparison Tables

### API Structure Comparison

| Feature | OpenAI | Claude | Gemini |
|---------|--------|--------|--------|
| **Base URL** | api.openai.com | api.anthropic.com | generativelanguage.googleapis.com |
| **Auth Header** | `Authorization: Bearer` | `x-api-key` | URL param or OAuth |
| **Messages Key** | `messages` | `messages` | `contents` |
| **Content Format** | `content: string` | `content: string/array` | `parts: array` |
| **System Message** | In messages | Separate `system` | `systemInstruction` |
| **Max Tokens** | `max_tokens` | `max_tokens` (required) | `maxOutputTokens` |
| **Version Header** | None | `anthropic-version` | None |

### Tool Calling Comparison

| Feature | OpenAI | Claude | Gemini |
|---------|--------|--------|--------|
| **Schema Key** | `parameters` | `input_schema` | `parameters` |
| **Tool Wrapper** | `{type: "function", function: {...}}` | Direct object | `{function_declarations: [...]}` |
| **Tool Choice** | `tool_choice` | `tool_choice` | `tool_config.function_calling_config.mode` |
| **Force Tool** | `{type: "function", function: {name}}` | `{type: "tool", name}` | `allowed_function_names` |
| **Tool Call ID** | `call_xxx` | `toolu_xxx` | N/A (use name) |
| **Result Role** | `role: "tool"` | `role: "user"` + `tool_result` | `role: "user"` + `functionResponse` |
| **Parallel Calls** | Yes | Yes | Yes |

### Vision Comparison

| Feature | OpenAI | Claude | Gemini |
|---------|--------|--------|--------|
| **Image in Content** | `image_url` | `image` | `inline_data` |
| **URL Support** | Yes | Yes | Via upload |
| **Base64 Support** | Yes | Yes | Yes |
| **Video Support** | No | No | Yes |
| **Audio Support** | Whisper API | No | Yes |
| **PDF Support** | No | Yes (via images) | Yes |
| **Detail Control** | `low/high/auto` | No | No |

### Model Capabilities

| Capability | OpenAI | Claude | Gemini |
|------------|--------|--------|--------|
| **Max Context** | 128K-200K | 200K | 2M |
| **Vision** | Yes | Yes | Yes |
| **Video** | No | No | Yes |
| **Audio** | Whisper | No | Yes |
| **Code Execution** | No | No | Yes |
| **Extended Thinking** | o1 models | Yes | Yes (thinking model) |
| **Structured Output** | Yes | Via tools | Yes |
| **Prompt Caching** | No | Yes | Yes |

---

## Agent Loop Patterns

### Basic Agent Loop

```python
async def agent_loop(query: str, tools: list, max_iterations: int = 10):
    messages = [{"role": "user", "content": query}]

    for i in range(max_iterations):
        # 1. Call LLM with tools
        response = await llm.invoke(messages, tools=tools)

        # 2. Check if tool calls were made
        tool_calls = extract_tool_calls(response)

        if not tool_calls:
            # No tool calls = final answer
            return extract_content(response)

        # 3. Add assistant message with tool calls
        messages.append(format_assistant_message(response))

        # 4. Execute tools and add results
        for tool_call in tool_calls:
            result = await execute_tool(tool_call)
            messages.append(format_tool_result(tool_call.id, result))

    raise Exception("Max iterations reached")
```

### Provider-Specific Tool Result Formatting

```python
def format_tool_result(provider: str, tool_call_id: str, result: str):
    if provider == "openai":
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result
        }
    elif provider == "anthropic":
        return {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": result
            }]
        }
    elif provider == "google":
        return {
            "role": "user",
            "parts": [{
                "functionResponse": {
                    "name": tool_call_id,  # Gemini uses name, not ID
                    "response": json.loads(result)
                }
            }]
        }
```

### ReAct Pattern

```python
REACT_PROMPT = """Answer using this format:

Thought: [your reasoning]
Action: tool_name
Action Input: {"param": "value"}
Observation: [tool result]
... (repeat as needed)
Thought: I have enough information
Final Answer: [your answer]
"""

async def react_agent(query: str, tools: list):
    messages = [
        {"role": "system", "content": REACT_PROMPT},
        {"role": "user", "content": query}
    ]

    while True:
        response = await llm.invoke(messages, tools=tools)

        if "Final Answer:" in response.content:
            return parse_final_answer(response.content)

        # Continue with tool execution...
```

---

## Best Practices

### 1. Tool Design

```python
# Good: Clear, specific description with examples
{
    "name": "search_products",
    "description": "Search for products by name, category, or price range. Returns up to 10 matching products with name, price, and availability.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (product name or keywords)"
            },
            "category": {
                "type": "string",
                "enum": ["electronics", "clothing", "home"],
                "description": "Filter by category"
            },
            "max_price": {
                "type": "number",
                "description": "Maximum price in USD"
            }
        },
        "required": ["query"]
    }
}

# Bad: Vague, no context
{
    "name": "search",
    "description": "Search for stuff",
    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}
}
```

### 2. Error Handling

```python
async def execute_tool_safely(tool_call, timeout=30):
    try:
        result = await asyncio.wait_for(
            execute_tool(tool_call),
            timeout=timeout
        )
        return {"success": True, "result": result}
    except asyncio.TimeoutError:
        return {"success": False, "error": f"Tool timed out after {timeout}s"}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

### 3. Rate Limit Handling

```python
import time
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5)
)
async def call_llm_with_retry(messages, **kwargs):
    return await client.chat.completions.create(
        messages=messages,
        **kwargs
    )
```

### 4. Token Management

```python
def trim_messages(messages: list, max_tokens: int, tokenizer):
    """Keep recent messages within token limit."""
    total = 0
    trimmed = []

    # Always keep system message
    system_msg = messages[0] if messages[0]["role"] == "system" else None
    other_msgs = messages[1:] if system_msg else messages

    for msg in reversed(other_msgs):
        tokens = len(tokenizer.encode(str(msg["content"])))
        if total + tokens > max_tokens:
            break
        trimmed.insert(0, msg)
        total += tokens

    if system_msg:
        trimmed.insert(0, system_msg)

    return trimmed
```

### 5. Caching

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_embedding(text: str) -> list:
    """Cache embeddings to reduce API calls."""
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

def cache_key(messages: list) -> str:
    """Generate cache key for conversation."""
    content = json.dumps(messages, sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()
```

---

## Rate Limits & Pricing

### OpenAI Rate Limits (Tier 1)

| Model | RPM | TPM | RPD |
|-------|-----|-----|-----|
| gpt-4o | 500 | 30,000 | 10,000 |
| gpt-4o-mini | 500 | 200,000 | 10,000 |
| gpt-4-turbo | 500 | 30,000 | 10,000 |
| o1 | 500 | 30,000 | 10,000 |

### Claude Rate Limits (Tier 1)

| Model | RPM | TPM (Input) | TPM (Output) |
|-------|-----|-------------|--------------|
| Claude Opus | 50 | 20,000 | 4,000 |
| Claude Sonnet | 50 | 40,000 | 8,000 |
| Claude Haiku | 50 | 50,000 | 10,000 |

### Gemini Rate Limits (Free Tier)

| Model | RPM | TPM | RPD |
|-------|-----|-----|-----|
| gemini-1.5-pro | 2 | 32,000 | 50 |
| gemini-1.5-flash | 15 | 1,000,000 | 1,500 |
| gemini-2.0-flash | 15 | 1,000,000 | 1,500 |

### Pricing Summary (per 1M tokens)

| Provider | Model | Input | Output |
|----------|-------|-------|--------|
| **OpenAI** | gpt-4o | $2.50 | $10.00 |
| | gpt-4o-mini | $0.15 | $0.60 |
| | o1 | $15.00 | $60.00 |
| **Anthropic** | claude-opus-4 | $15.00 | $75.00 |
| | claude-sonnet-4 | $3.00 | $15.00 |
| | claude-3.5-haiku | $0.80 | $4.00 |
| **Google** | gemini-1.5-pro | $1.25 | $5.00 |
| | gemini-1.5-flash | $0.075 | $0.30 |
| | gemini-2.0-flash | $0.10 | $0.40 |

---

## References

### Official Documentation
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic API Reference](https://docs.anthropic.com/en/api)
- [Anthropic Tool Use Documentation](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [Google Gemini API Reference](https://ai.google.dev/api)
- [Google Gemini Function Calling](https://ai.google.dev/gemini-api/docs/function-calling)

### SDKs
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python)
- [Google Generative AI Python SDK](https://github.com/google/generative-ai-python)

### Tools & Frameworks
- [LangChain](https://python.langchain.com/)
- [LlamaIndex](https://docs.llamaindex.ai/)
- [Instructor](https://github.com/jxnl/instructor) - Structured outputs
- [LiteLLM](https://github.com/BerriAI/litellm) - Unified API

---

*Last updated: January 2025*
