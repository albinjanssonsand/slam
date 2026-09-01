---
name: implement-issue
description: Fetch a GitHub issue from this repo, implement it on a new branch, self-review the diff, and run whatever validation exists. Use when the user says "implement issue #N", or similar.
---

# Implement Issue

Argument: an issue number or URL (e.g. `2` or `https://github.com/albinjanssonsand/slam/issues/2`).

1. **Fetch the issue.** `gh issue view <number> --repo albinjanssonsand/slam --json number,title,body,labels,comments`.
   If `gh` fails (not installed / not authenticated), stop and tell the user — don't guess at the issue content.

2. **Check working tree state.** `git status`. If there's uncommitted work, ask the user what to do with it — don't discard anything silently.

3. **Branch.** `git fetch origin`, then `git checkout -b issue-<number>-<short-kebab-slug-of-title> origin/main`. Never implement directly on `main`.

4. **Implement.** Read the issue body and comments as the spec. Explore the relevant code before writing anything. Make the minimal change the issue actually asks for — no drive-by refactors, no speculative abstractions. If there is a design choice to be made, or conflicts in the issue, ask the user. Match existing code style.

5. **Self-review.** Before calling it done, run the `code-review` skill against the diff and address anything it flags.

6. **Validate.** This repo has no automated test suite as of now. If the issue itself adds tests, run them (`pytest`). Otherwise, sanity-check the change by actually running the affected script/module (e.g. against a sample under `datasets/` or `recordings/`) and say plainly what you ran and what you observed — don't claim "tests pass" when none exist.

7. **Report, don't push.** Summarize what changed, what was validated and how, and stop. Ask the user before `git push` or opening a PR — those are visible to others and not implied by "implement the issue."
