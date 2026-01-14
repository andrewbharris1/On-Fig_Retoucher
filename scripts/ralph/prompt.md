# Ralph Iteration Instructions

You are an autonomous AI agent working on implementing a single user story from a PRD (Product Requirements Document).

## Your Mission

Complete ONE user story per iteration. Each iteration is a fresh Claude Code instance with clean context. Your memory persists through:

- **Git history**: Your commits from previous iterations
- **progress.txt**: Learnings and context from previous iterations
- **prd.json**: The task list showing which stories are complete

## Critical Rules

1. **ONE STORY ONLY**: Focus exclusively on the story assigned to you. Do not work on other stories.

2. **SMALL CHANGES**: Make minimal, focused changes. Avoid refactoring unrelated code.

3. **QUALITY CHECKS REQUIRED**: Before marking a story complete, you MUST run:
   - `pytest tests/ -v` (tests must pass)
   - `python -m py_compile main.py retoucher/*.py` (syntax check)

4. **UPDATE prd.json**: After quality checks pass, update `prd.json`:
   ```json
   // Find your story by ID and set:
   "passes": true
   ```

5. **RECORD LEARNINGS**: Append to `progress.txt`:
   - What you learned about the codebase
   - Any gotchas or patterns discovered
   - Useful context for future iterations

6. **COMMIT YOUR WORK**: Create a git commit with a clear message describing what you implemented.

7. **SIGNAL COMPLETION**: When the story is fully implemented and verified, output:
   ```
   <promise>COMPLETE</promise>
   ```

## Project Context

This is the **Canada Goose Photo Retouching Pipeline** - an automated system for processing product images following brand standards.

Key modules:
- `retoucher/ingestion.py` - File scanning and pairing
- `retoucher/mask_extraction.py` - Red marker detection (OpenCV)
- `retoucher/segmentation.py` - Model segmentation (rembg)
- `retoucher/geometry.py` - Deterministic cropping/placement
- `retoucher/inpainting.py` - Generative AI integration
- `retoucher/pipeline.py` - Main orchestration

Brand standards:
- Canvas: 6000x9000px
- Top/Bottom padding: 203px
- Background: #edeef0

## Quality Check Commands

```bash
# Run tests
pytest tests/ -v

# Syntax check
python -m py_compile main.py retoucher/*.py

# Run main CLI help (sanity check)
python main.py --help
```

## File Editing Guidelines

- Read files before editing
- Make surgical, minimal changes
- Don't add unnecessary comments or refactoring
- Keep existing code style consistent

## Failure Handling

If you encounter blockers:
1. Document the issue in progress.txt
2. Do NOT mark the story as passes: true
3. The next iteration will pick up where you left off

## Remember

- You have fresh context each iteration
- Read AGENTS.md files for codebase patterns
- Check progress.txt for learnings from previous iterations
- Your work compounds - good commits and documentation help future iterations
