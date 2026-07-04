# Delegation Guide — running IMPLEMENTATION_PLAN.md with a cheap implementer + Fable reviewer

> How to execute the milestones in `IMPLEMENTATION_PLAN.md` using Sonnet/Opus sessions for
> implementation and a Fable session for review. M1 was run as a subagent directly from the
> planning session; this guide covers M2 onward.

## The loop (per milestone)

```
1. IMPLEMENT   new Claude Code session, cheap model, one milestone, ends in a PR
2. REVIEW      Fable session reviews the PR against the plan
3. FIX         implementer session addresses review findings, pushes
4. MERGE       you merge the PR to develop
5. NEXT        repeat with the next milestone
```

Git is the handoff medium. The implementer never talks to the reviewer directly — the PR
diff plus `IMPLEMENTATION_PLAN.md` is the entire contract.

## 1. Implementer session

Start a fresh session per milestone (fresh = it must rely on the plan, not on chat memory):

```bash
cd ~/Projects/gfn-churn-prediction
claude --model sonnet        # or --model opus for the hard milestones (see table)
```

Paste this prompt, filling in the milestone ID:

```
Read CLAUDE.md and IMPLEMENTATION_PLAN.md in full before doing anything.
Execute milestone M<X> ONLY. Do not start any other milestone.

Rules:
- Follow the "Global rules (apply to every milestone)" section of IMPLEMENTATION_PLAN.md exactly.
- Do not modify IMPLEMENTATION_PLAN.md or DEVELOPMENT_PLAN.md, and do not change any decision
  recorded in PROJECT_PLAN.md. You MAY tick progress checkboxes and add the dated decision-log
  rows that the milestone explicitly tells you to add.
- Work on the branch named in the milestone, created from develop. Use conventional commits.
- Stop at every CHECKPOINT in the milestone and wait for my answer before continuing.
- If a step is ambiguous, or fails twice in a row, stop and ask me. Do not improvise around
  the plan or "fix" things the milestone doesn't mention.
- Run the milestone's Verify block before opening the PR and paste its full output to me.
- Open a PR to develop when Verify passes. In the PR description: the milestone ID, what was
  done, the Verify output, and any acceptance bands the milestone pre-registered. Then stop.
```

Answer checkpoints yourself in that session. For CHECKPOINT 2 (M4 hardening parameters):
copy the implementer's proposal into your Fable review session for a second opinion before
approving — that decision shapes every downstream metric.

## 2. Reviewer session (Fable)

After the PR opens, in a Fable session (any — project memory and the plan files carry
the context):

```
Review PR #<N>, which implements IMPLEMENTATION_PLAN.md milestone M<X>.
Check, in this order:
1. Plan fidelity — every step of M<X> done, nothing outside its scope touched.
2. Global-rules violations — persona leakage into features, week arithmetic on raw
   0-indexed counters, data/models/mlruns files committed, credentials.
3. Correctness bugs in the diff.
4. Whether the Verify output in the PR description actually demonstrates the
   milestone's acceptance criteria.
Then run /review <N>.
```

Escalate to the billed multi-agent review for the high-stakes PRs only —
type `/code-review ultra <PR#>` yourself (only you can trigger it): recommended for
M4 (data regen), M6 (LSTM), M7 (ensemble), M10 (AWS).

Send review findings back to the implementer session verbatim:

```
Reviewer found these issues in the PR — fix each one, or explain why it is not an issue.
Push the fixes to the same branch, rerun the Verify block, and paste the output.
<paste findings>
```

Merge yourself once the reviewer is satisfied. Then update the Fable session with one line
("M<X> merged") so its next review has current state.

## 3. Milestone → model → review level

| Milestone | Model | Review | Notes |
|---|---|---|---|
| M1 churn-gap investigation | (done — Sonnet subagent from planning session) | Fable inline | read-only |
| M2 tests + CI + README sync | Sonnet | `/review` | mechanical, fully specified |
| M3 MLflow | Sonnet | `/review` | small |
| M4 harden data + regen + v2 | **Opus** | **ultra** | CP2 before code, CP3 after metrics; bounce CP2 params past Fable |
| M5 SHAP | Sonnet | `/review` | |
| M6 LSTM | **Opus** | **ultra** | training-loop correctness is subtle |
| M7 ensemble | Sonnet | **ultra** | small code, high-stakes conclusion (pre-registered criterion) |
| M8 paths/CLI/validate | Sonnet | `/review` | |
| M9 full-scale run | **Opus** | Fable inline (no PR unless code changed) | Spark tuning judgment; asks before overwriting data/raw |
| M10 AWS | **Opus** | **ultra** | CP5 first: account, region, budget, endpoint type |
| M11 MLOps | **Opus** | `/review` per PR | may split into 2–3 PRs |

## 4. Failure modes to watch for

- **Implementer drifts out of scope** ("while I was here I also refactored…") — reviewer
  checks plan fidelity first for exactly this; reject out-of-scope changes, don't absorb them.
- **Implementer edits the plan to match what it built** — the prompt forbids it; reviewer
  should `git diff develop -- IMPLEMENTATION_PLAN.md DEVELOPMENT_PLAN.md` and expect empty.
- **Verify skipped or paraphrased** — require pasted raw output, not a summary.
- **Checkpoint answered by the implementer itself** — checkpoints are yours; if a PR appears
  containing checkpoint-gated work you never approved, that's a reject.
- **Spark-container steps** (M4 step 4, M9): the implementer may not be able to start Docker
  from its sandbox — the plan tells it to stop and ask you to run the container command; do that
  manually rather than letting it work around the container.
