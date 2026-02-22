# 更新日志

本项目的更新日志遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 规范，
版本号遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Changed
- 待记录

## [0.1.6] - 2026-02-22

### Fixed
- 修复 Codex 场景下 `prompt_cache_key` 超长导致上游 400（`string_above_max_length`）进而中断对话的问题。
- `prompt_cache_key` 生成策略改为固定哈希格式 `astrbot:pc:v1:<digest>`，确保 ASCII 且长度不超过 64。
- 当上游明确返回 `param=prompt_cache_key` 错误时，自动移除该字段并重试一次，避免任务流中断。

### Changed
- 补充并强化 Codex 适配测试，覆盖缓存键稳定性、长度约束与错误自愈重试场景。

