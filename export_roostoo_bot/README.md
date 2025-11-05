Roostoo Bot - Exported Minimal Runtime

This folder contains a minimal runtime export of the Roostoo trading bot prepared for adding into a repository and pushing into GitHub.

Contents
- `roostoo_bot_template/`
  - `bot.py` - main bot loop and a `--dry-run` single-iteration mode
  - `strategy.py` - indicators and signal logic (EMA, RSI, ATR, ADX)
  - `roostoo_api.py` - minimal signed Roostoo API client
  - `metrics.py` - small performance helpers
  - `utils/logger.py` - tiny logging helper
  - `__init__.py` - package exports

Add-on files
- `requirements.txt` - minimal Python dependencies
- `.gitignore` - recommended ignores

Purpose
This export is intended to be merged into your target repository (for example, `https://github.com/aaqwesas/Web3-Quant-Hackathon-2025.git`). It contains only runtime code necessary to run the bot and the minimal helper modules.

How to add this export to your GitHub repo (PowerShell, run from the repository root)
1. Copy the `export_roostoo_bot/roostoo_bot_template` directory into your repo (preserve path). For example:

```powershell
# from the repo root
robocopy "C:\Users\Bernardinus Fernando\Documents\roostoo_bot_template\export_roostoo_bot\roostoo_bot_template" ".\roostoo_bot_template" /E
```

2. Create a branch, commit and push:

```powershell
git checkout -b add/roostoo-bot-export
git add roostoo_bot_template README.md requirements.txt .gitignore
git commit -m "Add minimal Roostoo bot runtime export"
git push origin add/roostoo-bot-export
```

3. Open a Pull Request on GitHub to merge `add/roostoo-bot-export` into your target branch (e.g., `main`).

If you'd like, I can prepare a patch or run the git commands for you — I will need either:
- a) the repository cloned into this workspace and configured with push access, or
- b) a GitHub personal access token (PAT) with repo scope so I can push remotely (please only share tokens via a secure channel and rotate them after use). If you prefer not to share credentials, follow the steps above locally or ask me to generate a ready-to-apply patch.

Notes
- The bot expects a `config.yaml` or `config.local.yaml` in the repository root with keys for `roostoo` and `bot`. See the original project for examples.
- Start in `--dry-run` mode and test against a sandbox / small positions before enabling live trading.

Contact
Reply here with how you'd like me to proceed: produce a ZIP/patch, or (if you want me to push) provide one of the options above (cloned repo in workspace OR a PAT).