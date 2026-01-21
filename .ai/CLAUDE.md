# Claude Collaboration Notes

## Project Context
Working on sp_tokens rewrite for flexible tokenizer generation (hierarchical group tokens, configurable layers, overlap tokens). See `.ai/sp_tokens_rewrite_plan.md` for details.

When the user let you do something, first talk with the user patiently. Make sure the user has a clear plan and design, then impl, no hurry.

## Notes
- User runs commands on VM, so just provide the `python -m` command instead of running directly.
- some dump script could be run locally, use `uv run -m` command