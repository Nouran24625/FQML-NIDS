# Instructions for Claude working on FQML-NIDS

Read `CONTEXT.md` first: it is the team's single source of truth for status, owners, decisions and deadlines. Then read `docs/DECISIONS.md` and `docs/INTERFACES.md` before changing anything they cover.

## Rules
- Do not add "Co-Authored-By: Claude" (or any Claude attribution) to commits, PR descriptions or files.
- Everything runs in Docker: `docker compose -f docker/compose.yml exec dev <command>` from the repo root.
- Before editing an existing file, read its current contents; do not guess them.
- Never ask for, print or commit tokens or keys.
- New libraries go in `requirements.txt` via a PR; never hand-edit `requirements/lock.txt`.
- Interface changes: one PR titled "Interface change: …" with a version bump of `docs/INTERFACES.md`.
- Every PR needs a reviewer from a different area; nobody merges their own PR.
- Team plans name each member (Nouran, Ganna, Kareem, Sheghaf); never address the reader as "you".
- Propose a design change only if a checked fact shows it raises the value of the results; time or compute cost is not a reason against it.
- No "first" claims, no quantum-advantage claims, no "post-quantum secured federation" unless built and measured.
- At the end of a session that changes status, a decision or a deadline, update `CONTEXT.md` (replace stale lines, bump the dates at the top).
