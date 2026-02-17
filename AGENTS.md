# Repository Guidelines

## Overview
AstrBot plugin adding an OpenAI Responses API provider: `openai_responses_plugin`.

## Project Structure
- `main.py`: entry; registers provider + WebUI template.
- `provider_responses.py`: adapter (HTTP, SSE parsing, tools, usage).
- `register_provider.py`: idempotent register/unregister helpers.
- `metadata.yaml`: plugin metadata.
- `tests/`: `pytest` unit tests.

## Commands
- Tests: `python -m pytest -q`
- Format/lint (optional): `ruff format .` / `ruff check .`
- Debug in AstrBot: install under `data/plugins/astrbot_plugin_openai_responses/`; restart AstrBot (process-global registry).

## Style
- Python: 4-space indent; keep typing style consistent (`from __future__ import annotations`).
- New comments: English.
- Paths: prefer `pathlib.Path`; use `astrbot.core.utils.path_utils` for data/tmp dirs.

## Testing
- Files: `tests/test_*.py`; async via `@pytest.mark.asyncio`.
- Cover: schema conversion, stream delta accumulation, tool-call args, token usage.

## Commit & Pull Request Guidelines
- No Git history yet; use Conventional Commits (`feat: ...`, `fix: ...`, `chore: ...`).
- PRs (prefer `gh`): English title/description; include motivation, scope, test output, and screenshots for config/UI changes.

## Security
- Don’t commit API keys or proxy creds; use `sk-...` placeholders.
