---
name: code-review
description: Review the current diff, or a PR number/branch/path target, for correctness bugs and reuse/simplification/efficiency cleanups. Runs 1 reviewer agent by default; scales to 2-3 only for large, complex, or security-sensitive changes, or when explicitly asked for a deeper/parallel review. Pass --comment to post findings as inline PR comments, or --fix to apply findings to the working tree after the review.
---

# Code Review

Project-level override of the built-in `code-review` skill, scoped to this
repo. The only thing this changes vs. the built-in default is **how many
reviewer agents run** — everything else (what counts as a finding, how
findings are reported) works the same.

**Why this override exists:** the built-in skill fans out into 8 parallel
finder agents by default (one per fixed "angle": correctness, simplification,
reuse/duplication, efficiency, altitude/architecture, cross-file tracing,
line-by-line diff, CLAUDE.md conventions), driven by this account's
`effortLevel: "high"` setting in `~/.claude/settings.json`. That's a lot of
token/time cost for a small, everyday diff. This override makes 1 agent the
default and only scales up when the diff actually warrants it.

## 1. Determine the review target

- No argument, or `--comment`/`--fix` only: review the current working diff
  (`git diff` against the merge-base of the current branch and `main`, plus
  any staged/unstaged changes — same scope `git status`/`git diff` would show).
- A PR number (e.g. `123`) or a `owner/repo#123` reference: `gh pr diff
  <number>` (add `--repo` if the reference includes one).
- A branch name: `git diff main...<branch>`.
- A file/directory path: review just that path's current diff.
- The literal argument `ultra`: **do not attempt to replicate this locally.**
  `ultra` is a separate, explicitly user-triggered and billed cloud review
  (`/code-review ultra`, or `ultrareview`) that launches outside this
  session. Tell the user it needs to be launched from the standard
  `/code-review ultra` command, not this skill.

## 2. Decide reviewer count

Start from **1 reviewer agent**. Escalate to **2** or **3** only if one or
more of these hold — check them in order and stop at the first match that
applies, using its suggested count:

1. **Explicit ask.** The user's own words say to go deeper — "thorough",
   "deep review", "use N reviewers", "parallel review", "multiple angles",
   or they pass an explicit level like `high`/`xhigh`/`max`. → **3**.
2. **Security-sensitive.** The diff touches authentication, authorization,
   secrets/credentials, cryptography, network request handling, input
   parsing/deserialization of untrusted data, subprocess/shell execution,
   SQL, or file-path/URL construction from user input (grep the diff's
   changed files for names/paths like `auth`, `login`, `token`, `secret`,
   `password`, `crypto`, `jwt`, `sql`, `subprocess`, `eval`, `exec`,
   `deserialize`, or inspect the actual diff content when filenames alone
   are inconclusive). → **3**.
3. **Large.** The diff changes more than ~400 total lines (`git diff
   --stat`'s insertions+deletions) or touches more than ~8 files. → **2**.
4. **Complex.** The diff spans multiple interacting modules/subsystems in a
   way that a single reviewer plausibly can't hold in context at once (e.g.
   touches both a core data model/schema and every one of its consumers, or
   changes a shared interface plus all implementers) — judge this from
   actually skimming the diff's file list and shape, not just its size.
   → **2**.

If none apply, stay at **1**. State the chosen count and which rule (if any)
triggered it before launching agents, so the user can see why.

## 3. Run the review

Fetch the diff/target content first (`git diff`, `gh pr diff`, etc.) so you
know its real shape before deciding step 2 above, if you haven't already.

- **1 reviewer:** Do the review yourself, inline, in this conversation — no
  need to spawn a subagent for a single reviewer; you have full context
  already. Read every changed file's relevant surrounding context (not just
  the diff hunks) before flagging anything. Look for:
  - **Correctness bugs**: logic errors, off-by-one, wrong edge-case handling,
    incorrect assumptions about types/shapes/null-ness, concurrency issues.
  - **Reuse/simplification/efficiency cleanups**: duplicated logic that
    should share a helper, unnecessary complexity, redundant work, dead code.
  Rank findings most-severe-first. Verify each one before including it (trace
  through an actual failing input/scenario) — don't report a "maybe."

- **2-3 reviewers:** Launch that many `Agent` tool calls (subagent_type
  `general-purpose`), **in a single message** so they run in parallel. Give
  each one the full diff/target and the same two finding categories above
  (correctness; reuse/simplification/efficiency), but assign each a distinct
  focus angle so they don't produce redundant output — pick angles that fit
  the actual diff, e.g.:
  - Reviewer 1: correctness — line-by-line logic/edge-case audit.
  - Reviewer 2: correctness — cross-file consistency (does this change
    honor every caller's/consumer's assumptions? did it silently remove or
    change behavior something else depends on?).
  - Reviewer 3 (only when count is 3): reuse/simplification/efficiency, plus
    architecture-fit ("does this match how the rest of the codebase solves
    similar problems?").
  Each agent should report back in the same structured form (file, summary,
  concrete failure scenario) rather than prose, so findings can be merged.

## 4. Dedupe, verify, and report

- Merge findings across reviewers (when >1 ran); collapse near-duplicates
  reported by more than one reviewer into a single finding.
- Drop anything you can't back with a concrete failure scenario (specific
  input/state → wrong output/crash) — a vague "this could be an issue" isn't
  a finding.
- Rank the survivors most-severe-first.
- Report with the `ReportFindings` tool, one call, findings ranked
  most-severe-first (empty array if nothing survived verification). Do not
  also print the findings as plain text when this tool is available.

## 5. Optional flags

- `--fix`: after reporting, apply the reported findings to the working tree,
  then re-report with each finding's `outcome` set (`fixed`/`skipped`/
  `no_change_needed`).
- `--comment`: if the target is a real GitHub PR (a number/URL, not the
  local working diff), post the findings as inline PR review comments via
  `gh`. Confirm with the user first if it's not already clear they want
  this to be visible to others on the PR.
