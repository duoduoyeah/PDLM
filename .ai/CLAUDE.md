# Claude Collaboration Notes


## Notes
- User runs commands on VM, so just provide the `python -m` command instead of running directly.
- Only dump scripts (in `scripts/dump/` or `group_tokenizer/dump.py`) can be run locally with `uv run -m` command.
- Do NOT run other scripts (training, building, etc.) locally - just write them and let user run on VM.
- When the user asks to do something, first discuss and clarify the design before implementing.

