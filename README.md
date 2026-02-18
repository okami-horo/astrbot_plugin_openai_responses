# astrbot_plugin_openai_responses

这是一个为 AstrBot 提供 **OpenAI Responses API**（`/v1/responses`）兼容能力的实验性插件。

本插件会注册一个新的 Provider `type`：`openai_responses_plugin`，用于避免与上游/其他实现发生命名冲突。

## 安装

1. 将本插件仓库安装到 AstrBot（插件市场 / WebUI / 指令均可）。
2. 安装/更新后建议 **重启 AstrBot**（Provider 注册属于进程级全局状态，重启最稳）。

## 配置（WebUI）

1) 在「模型提供商」→「提供商源」中新增一个 provider source：

- `id`: 自定义一个唯一 ID（例如 `openai_responses_p`）
- 类型选择：`OpenAI (Responses, Plugin)`（对应 `type=openai_responses_plugin`）
- `key`: OpenAI API Key 列表（支持多个 Key）
- `api_base`: 默认 `https://api.openai.com/v1`
- `timeout`: 默认 `120`（秒）
- `proxy`: 可选，例如 `http://127.0.0.1:7890`
- `custom_headers`: 可选，自定义请求头
- `reasoning_effort`: 可选，`low/medium/high/xhigh`（不填则不传）
- `custom_extra_body`: 可选，透传到 Responses 请求 body 的额外字段
- `tool_fallback_enabled`: 默认 `true`，开启“伪调用检测 + 工具重试”兜底
- `tool_fallback_mode`: 默认 `parse_then_retry`，可选 `parse_then_retry/retry_only/parse_only`
- `tool_fallback_retry_attempts`: 默认 `1`，伪调用时最多重试次数（建议 `0~3`）
- `tool_fallback_force_tool_choice`: 默认 `required`，重试时写入 `tool_choice`
- `tool_fallback_stream_buffer`: 默认 `true`，流式下先缓冲再输出，避免伪调用文本泄露给用户

2) 在「模型提供商」→「服务提供商」中新增一个 provider，并指向上面创建的 provider source：

- `provider_source_id`: 填上一步创建的 `id`
- `model`: 例如 `gpt-4o-mini`
- `id`: 自定义（例如 `openai_responses_p/gpt-4o-mini`）

最后将 `provider_settings.default_provider_id` 指向你刚创建的 provider `id`。

### 进阶：对应到 `cmd_config.json`（provider_sources + provider）

如果你使用的是两段式配置（`provider_sources` + `provider`），可以参考：

```json
{
  "provider_sources": [
    {
      "id": "openai_responses_plugin",
      "type": "openai_responses_plugin",
      "provider": "openai",
      "provider_type": "chat_completion",
      "enable": true,
      "key": ["sk-..."],
      "api_base": "https://api.openai.com/v1",
      "timeout": 120,
      "proxy": "",
      "custom_headers": {},
      "reasoning_effort": "",
      "custom_extra_body": {},
      "tool_fallback_enabled": true,
      "tool_fallback_mode": "parse_then_retry",
      "tool_fallback_retry_attempts": 1,
      "tool_fallback_force_tool_choice": "required",
      "tool_fallback_stream_buffer": true
    }
  ],
  "provider": [
    {
      "id": "openai_responses_plugin/gpt-4o-mini",
      "provider_source_id": "openai_responses_plugin",
      "model": "gpt-4o-mini",
      "enable": true
    }
  ]
}
```

## 工具调用兜底说明

- 触发条件：当前轮有工具可用，且模型返回文本里出现类似 `assistant to=functions.xxx` 的“伪调用”。
- 默认策略：先尝试在本地解析伪调用文本；解析失败再用 `tool_choice=required` 重试一次。
- `tool_fallback_mode`：
  - `parse_then_retry`：先解析，再重试（默认）
  - `retry_only`：不做本地解析，只做强制工具重试
  - `parse_only`：只做本地解析，不重试
- 流式注意：`tool_fallback_stream_buffer=true` 时，插件会先缓冲该轮流式结果，确认无需重试后再输出，因此首字延迟可能略增，但可避免把伪调用文本直接发给用户。

## 已知限制

- 本插件不尝试覆盖/替换 AstrBot 内置的 OpenAI ChatCompletions Provider；它只提供一个新的 Provider `type`。
- 插件卸载/禁用后，如配置仍引用该 provider，建议重启并移除相关 provider 配置后再启动。
