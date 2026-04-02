# Copilot Instructions

## Beads Issue Tracker

This project uses **bd** (beads) for issue tracking. Run `bd prime` to see
full workflow context and commands.

FrontPulse uses the canonical local shared Dolt server root at
`C:\Users\cmb17\.beads\shared-server` on `127.0.0.1:3308`. The canonical
project database is `FP`, and the repo issue prefix must stay `FP`.

Only direct child databases under `C:\Users\cmb17\.beads\shared-server` are
valid. Do not create nested server roots such as
`C:\Users\cmb17\.beads\shared-server\dolt`, do not start repo-local Dolt
servers for beads, and do not create umbrella trackers from `D:\Git_Repos`.
There is no remote Dolt sync for beads in this workspace, so do not run
`bd dolt push` or `bd dolt pull`.

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

## Rules

- Use bd for ALL task tracking. Do NOT use TodoWrite, TaskCreate, or markdown
  TODO lists.
- Run `bd prime` for detailed command reference and session close protocol.
- Use `bd remember` for persistent knowledge. Do NOT use `MEMORY.md` files.
