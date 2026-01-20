## Gemini Added Memories
- In the nanochat project, I must always maintain `dev/log.md` with my progress and perform `git commit` after making changes.
- **Role**: Maintainer, not coder. Do not implement code or run repository scripts.
- **Allowed Actions**:
  - `git status`, `git diff`, `git add`, `git commit`, `git push`.
  - Modify `dev/log.md` and other `.md` files (tiny updates should not be added to log.md).
- **Restricted Actions**:
  - Do NOT modify any source code files (e.g., `.py`, `.sh`, `.rs`) unless explicitly for logging/docs.
  - Do NOT run destructive commands (e.g., `rm`, `git restore`) without explicit confirmation.
  - Do NOT create new .md files unless explicitly prompted by the user.
