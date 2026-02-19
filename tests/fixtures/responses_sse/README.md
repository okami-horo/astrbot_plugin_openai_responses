# Responses SSE 夹具（回归测试用）

本目录用于保存 **OpenAI Responses** 的 SSE（`text/event-stream`）事件序列夹具，便于在单元测试中稳定复现以下问题：

- **US1**：文本增量事件组合差异（`response.output_text.delta` vs `response.content_part.delta`），以及 `done/completed` 事件缺失/乱序导致的截断、重复、错序
- **US2**：工具调用事件（`function_call`）的交错/分片（arguments delta）累积与“不泄露”策略验证
- **US3**：SSE 内嵌 `error` / `response.failed` 等失败事件的解析与可诊断（脱敏）输出

## 夹具格式约定

夹具文件使用“原始 SSE 行文本”，即每行是从网络流中读到的一行（不含换行符），并用空行分隔事件：

```text
event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"Hello"}

data: [DONE]
```

说明：

- `event:` 行可选；若不存在，解析器应优先使用 `data.type` 推断事件类型
- 支持多行 `data:`（例如格式化 JSON 被拆成多行）
- 流结束以 `data: [DONE]` 表示

## 示例夹具

- `us1_mixed_text.sse`：混合文本事件（`output_text.delta` + `content_part.delta`）+ `completed` 最终对齐
- `us2_tool_calls_interleaved.sse`：工具调用 added + arguments delta/done + completed output 回填
- `us3_error_event.sse`：SSE 内嵌 `error` 事件（用于验证解析与错误分类/脱敏）

