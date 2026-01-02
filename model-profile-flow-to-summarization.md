# Model Profile Flow to SummarizationMiddleware

This document explains how model profiles are populated and passed to the `SummarizationMiddleware` in LangChain/LangGraph agents.

## Overview

The profile system allows LangChain to automatically configure intelligent token-based summarization based on each model's actual capabilities (like max input tokens), rather than using hardcoded values.

## The Complete Flow

### 1. Profile Definition (Hardcoded Data)

Model capabilities are defined in a registry file maintained by LangChain:

**File**: `langchain_anthropic/data/_profiles.py`

```python
_PROFILES: dict[str, dict[str, Any]] = {
    "claude-sonnet-4-5-20250929": {
        "max_input_tokens": 200000,
        "max_output_tokens": 64000,
        "image_inputs": True,
        "audio_inputs": False,
        "video_inputs": False,
        "reasoning_output": True,
        "tool_calling": True,
        "image_url_inputs": True,
        "pdf_inputs": True,
        "pdf_tool_message": True,
        "image_tool_message": True,
        "structured_output": False,
    },
    "claude-opus-4-5-20251101": {
        "max_input_tokens": 200000,
        "max_output_tokens": 64000,
        # ... other capabilities
    },
    # ... more models
}
```

**Note**: This file is auto-generated from the [models.dev](https://github.com/sst/models.dev) project (MIT License).

### 2. Profile Auto-Population

When you instantiate a chat model, the profile is automatically populated via a Pydantic validator:

**File**: `langchain_anthropic/chat_models.py:922-926`

```python
@model_validator(mode="after")
def _set_model_profile(self) -> Self:
    """Set model profile if not overridden."""
    if self.profile is None:
        self.profile = _get_default_model_profile(self.model)
    return self
```

**Helper function** (line 78-90):
```python
def _get_default_model_profile(model_name: str) -> ModelProfile:
    """Get the default profile for a model."""
    default = _MODEL_PROFILES.get(model_name)
    if default:
        return default.copy()
    return {}
```

**Example**:
```python
from langchain_anthropic import ChatAnthropic

# Profile is automatically populated!
model = ChatAnthropic(model="claude-sonnet-4-5-20250929")
print(model.profile)
# Output: {'max_input_tokens': 200000, 'max_output_tokens': 64000, ...}
```

### 3. Profile Usage in `create_deep_agent`

The DeepAgents library checks if the model has a profile to determine summarization strategy:

**File**: `deepagents/graph.py:101-111`

```python
if (
    model.profile is not None
    and isinstance(model.profile, dict)
    and "max_input_tokens" in model.profile
    and isinstance(model.profile["max_input_tokens"], int)
):
    # Smart fractional mode - adapts to model's actual capacity
    trigger = ("fraction", 0.85)  # Trigger at 85% of max tokens
    keep = ("fraction", 0.10)     # Keep 10% of max tokens after summarization
else:
    # Fallback mode - uses hardcoded values
    trigger = ("tokens", 170000)  # Trigger at 170K tokens
    keep = ("messages", 6)        # Keep 6 messages
```

This `trigger` and `keep` are then passed to `SummarizationMiddleware`:

```python
SummarizationMiddleware(
    model=model,
    trigger=trigger,  # ("fraction", 0.85) or ("tokens", 170000)
    keep=keep,        # ("fraction", 0.10) or ("messages", 6)
    trim_tokens_to_summarize=None,
)
```

### 4. SummarizationMiddleware Reads Profile

The middleware accesses `model.profile` to calculate actual token thresholds:

**File**: `langchain/agents/middleware/summarization.py:404-419`

```python
def _get_profile_limits(self) -> int | None:
    """Retrieve max input token limit from the model profile."""
    try:
        profile = self.model.profile
    except AttributeError:
        return None

    if not isinstance(profile, Mapping):
        return None

    max_input_tokens = profile.get("max_input_tokens")

    if not isinstance(max_input_tokens, int):
        return None

    return max_input_tokens
```

**Using the profile for fractional triggers** (line 328-340):

```python
if kind == "fraction":
    max_input_tokens = self._get_profile_limits()  # Returns 200000
    if max_input_tokens is None:
        continue
    threshold = int(max_input_tokens * value)      # 200000 * 0.85 = 170000
    if threshold <= 0:
        threshold = 1
    if total_tokens >= threshold:
        return True  # Trigger summarization!
```

**Using the profile for fractional keep** (line 361-366):

```python
if kind == "fraction":
    max_input_tokens = self._get_profile_limits()  # Returns 200000
    if max_input_tokens is None:
        return None
    target_token_count = int(max_input_tokens * value)  # 200000 * 0.10 = 20000
```

## Example: claude-sonnet-4-5-20250929

For this model with `max_input_tokens: 200000`:

- **Trigger threshold**: `200000 * 0.85 = 170,000 tokens`
  - Summarization kicks in when conversation reaches 170K tokens

- **Keep amount**: `200000 * 0.10 = 20,000 tokens`
  - After summarization, keep the most recent 20K tokens worth of messages

## Modes Comparison

| Mode | Profile Required | Trigger Config | Keep Config | Behavior |
|------|-----------------|----------------|-------------|----------|
| **Fractional** | ✅ Yes | `("fraction", 0.85)` | `("fraction", 0.10)` | Adapts to model's actual max tokens |
| **Absolute Tokens** | ❌ No | `("tokens", 170000)` | `("tokens", 20000)` | Fixed token counts regardless of model |
| **Message Count** | ❌ No | `("messages", 100)` | `("messages", 6)` | Fixed message counts |

## Benefits of Profile-Based Fractional Mode

1. **Model-aware**: Automatically adapts to different models' capabilities
2. **Future-proof**: New models with larger context windows work correctly without code changes
3. **Optimal usage**: Uses the model's full capacity (85%) before summarizing
4. **Consistent behavior**: Same percentage-based logic across all models

## Manual Profile Override

You can manually set or override the profile:

```python
from langchain_anthropic import ChatAnthropic

# Override profile for custom models or testing
model = ChatAnthropic(
    model="custom-model-name",
    profile={"max_input_tokens": 128000}
)
```

## Profile Property Location

The `profile` property is defined in the base class:

**File**: `langchain_core/language_models/chat_models.py:340`

```python
profile: ModelProfile | None = Field(default=None, exclude=True)
"""Profile detailing model capabilities.

!!! warning "Beta feature"
    This is a beta feature. The format of model profiles is subject to change.
"""
```

## Type Definitions

```python
from langchain_core.language_models import ModelProfile

# ModelProfile is a TypedDict with keys like:
# - max_input_tokens: int
# - max_output_tokens: int
# - image_inputs: bool
# - tool_calling: bool
# ... etc
```

## Summary

The profile flow is:

1. **Hardcoded data** → `_PROFILES` dictionary in `langchain_anthropic/data/_profiles.py`
2. **Auto-population** → `_set_model_profile()` validator sets `model.profile` during initialization
3. **Strategy selection** → `create_deep_agent()` checks profile existence to choose fractional vs absolute mode
4. **Runtime calculation** → `SummarizationMiddleware._get_profile_limits()` reads `model.profile` to compute thresholds

**Key takeaway**: The profile is automatically populated when you create a ChatAnthropic model, and the middleware intelligently uses it to adapt summarization behavior to each model's actual capabilities.

## Related Files

- Profile data: `langchain_anthropic/data/_profiles.py`
- Profile population: `langchain_anthropic/chat_models.py:922-926`
- Profile usage in agent: `deepagents/graph.py:101-111`
- Profile usage in middleware: `langchain/agents/middleware/summarization.py:404-419`
- Base profile definition: `langchain_core/language_models/chat_models.py:340`

## References

- LangChain Docs: https://docs.langchain.com/oss/python/langchain/models
- Models.dev (source data): https://github.com/sst/models.dev
