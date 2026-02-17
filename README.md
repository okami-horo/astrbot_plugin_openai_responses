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
      "custom_extra_body": {}
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

## 已知限制

- 本插件不尝试覆盖/替换 AstrBot 内置的 OpenAI ChatCompletions Provider；它只提供一个新的 Provider `type`。
- 插件卸载/禁用后，如配置仍引用该 provider，建议重启并移除相关 provider 配置后再启动。
