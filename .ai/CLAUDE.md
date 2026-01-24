# Claude Collaboration Notes

## Project Context
Working on Experiment C

Key docs in `.ai_private/`:
- `experiment_c.md` - Experiment C design
- `sp_tokens_rewrite_plan.md` - Group tokenizer architecture
- `pdlm_eval_design.md` - New eval system design
- `log.md` - Progress tracking

## Maintaining log.md

**DO add to log.md:**
- Completed tasks (mark as [Done])
- New planned work the user mentions

**DO NOT add to log.md:**
- Bug fixes
- Debug sessions
- Minor tweaks

## Notes
- User runs commands on VM, so just provide the `python -m` command instead of running directly.
- Only dump scripts (in `scripts/dump/` or `group_tokenizer/dump.py`) can be run locally with `uv run -m` command.
- Do NOT run other scripts (training, building, etc.) locally - just write them and let user run on VM.
- When the user asks to do something, first discuss and clarify the design before implementing.
