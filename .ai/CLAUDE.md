# Claude Collaboration Notes

## Project Context
Working on sp_tokens rewrite for flexible tokenizer generation (hierarchical group tokens, configurable layers, overlap tokens). See `.ai/sp_tokens_rewrite_plan.md` for details.

You need to maintain those .md file in `.ai` folder
When the user let you do something, first talk with the user patiently. Make sure the user has a clear plan and design, then impl, no hurry.

## Notes
- User runs commands on VM, so just provide the `python -m` command instead of running directly.
- Only dump scripts (in `scripts/dump/`) can be run locally with `uv run -m` command.
- Do NOT run other scripts (training, building, etc.) locally - just write them and let user run on VM.