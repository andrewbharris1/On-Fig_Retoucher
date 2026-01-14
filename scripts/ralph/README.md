# Ralph for Claude Code

An autonomous AI agent loop that runs Claude Code repeatedly until all PRD items are complete.

Based on [Geoffrey Huntley's Ralph pattern](https://ghuntley.com/ralph/).

## How It Works

1. **Fresh context each iteration** - Each loop spawns a new Claude Code instance
2. **Memory persists via files**:
   - `prd.json` - Task list with pass/fail status
   - `progress.txt` - Learnings from previous iterations
   - Git history - Commits from previous work
3. **One story per iteration** - Focus on completing one small task at a time
4. **Quality gates** - Tests must pass before marking complete

## Prerequisites

- Claude Code CLI installed and authenticated (`claude --version`)
- `jq` installed (`apt install jq` or `brew install jq`)

## Quick Start

1. **Create your PRD**:
   ```bash
   cp prd.json.example prd.json
   # Edit prd.json with your user stories
   ```

2. **Run Ralph**:
   ```bash
   ./scripts/ralph/ralph.sh
   ```

3. **Monitor progress**:
   ```bash
   # Check story status
   cat prd.json | jq '.userStories[] | {id, title, passes}'

   # View learnings
   cat progress.txt

   # Git history
   git log --oneline -10
   ```

## PRD Format

```json
{
  "name": "Feature Name",
  "branchName": "feature/my-feature",
  "userStories": [
    {
      "id": "STORY-001",
      "title": "Short title",
      "description": "What needs to be done",
      "priority": 1,
      "acceptanceCriteria": [
        "Criterion 1",
        "Criterion 2"
      ],
      "passes": false
    }
  ]
}
```

## Key Concepts

### Small Tasks

Each story should be completable in one context window. Break large tasks into smaller pieces.

**Good**:
- "Add validation to user input field"
- "Create database migration for new column"
- "Add unit tests for geometry module"

**Too big (split these)**:
- "Build the entire dashboard"
- "Implement authentication system"

### Learnings Compound

After each iteration, update:
- `progress.txt` - What you learned
- `AGENTS.md` - Codebase patterns discovered

### Quality Checks

Before marking complete, Claude Code runs:
```bash
pytest tests/ -v
python -m py_compile main.py retoucher/*.py
```

## Files

| File | Purpose |
|------|---------|
| `ralph.sh` | Main loop script |
| `prompt.md` | Instructions for each Claude instance |
| `prd.json` | User stories (you create this) |
| `prd.json.example` | Example PRD format |
| `progress.txt` | Learnings log (auto-created) |
| `AGENTS.md` | Codebase patterns |

## Customization

Edit `prompt.md` to:
- Add project-specific quality checks
- Include codebase conventions
- Add common gotchas for your stack

## Safety

By default, `ralph.sh` uses `--dangerously-skip-permissions` for unattended operation. For safer runs, edit `ralph.sh` and remove this flag - Claude will prompt for tool approvals.
