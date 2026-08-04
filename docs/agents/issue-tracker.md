# Issue tracker: GitHub

Issues and PRDs for this repository live as GitHub issues. Use the `gh` CLI for all operations.

## Conventions

- **Create an issue:** `gh issue create --title "..." --body "..."`.
- **Read an issue:** `gh issue view <number> --comments`, including labels.
- **List issues:** use `gh issue list` with the appropriate state and label filters.
- **Comment:** `gh issue comment <number> --body "..."`.
- **Apply or remove labels:** `gh issue edit <number> --add-label "..."` or `--remove-label "..."`.
- **Close:** `gh issue close <number> --comment "..."`.

Infer the repository from `git remote -v`; `gh` does this automatically inside the clone.

## Pull requests as a triage surface

**PRs as a request surface: no.** External pull requests do not enter the issue-triage queue.

## Skill publishing and retrieval

- When a skill says to publish to the issue tracker, create a GitHub issue.
- When a skill says to fetch a ticket, run `gh issue view <number> --comments`.

## Wayfinding operations

The map is one GitHub issue and its tickets are child issues.

- **Map:** create an issue labelled `wayfinder:map`. Its body contains Destination, Notes, Decisions so far, Not yet specified, and Out of scope.
- **Child ticket:** link an issue to the map as a GitHub sub-issue. If sub-issues are unavailable, add it to a task list in the map and put `Part of #<map>` at the top of the child body. Apply one of `wayfinder:research`, `wayfinder:prototype`, `wayfinder:grilling`, or `wayfinder:task`.
- **Blocking:** use GitHub's native issue dependencies. If dependencies are unavailable, put `Blocked by: #<n>, #<n>` at the top of the child body.
- **Frontier:** among the map's open child issues, the frontier is the ordered set with no open blockers and no assignee.
- **Claim:** assign the issue to the driving developer before beginning work.
- **Resolve:** post the answer as a resolution comment, close the issue, and append a one-line linked gist to the map's Decisions-so-far section.

## Wayfinding asset isolation

Each Wayfinder effort gets its own folder under `docs/wayfinding/`. The solid-first subtractive-modeling effort uses `docs/wayfinding/solid-first-subtractive-modeling/`. Do not place its assets in or modify `.scratch/transform-composition-proof/`; that is a separate active experiment.
