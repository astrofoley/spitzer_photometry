# Git commit and pull request (maintainer)

After editing, from the repository root:

```bash
# Use a descriptive branch name
git checkout -b feature/nominal-native-science-run-docs

git status
git add .gitignore README.md docs/NOMINAL_NATIVE_SCIENCE_RUN.md docs/GIT_AND_PULL_REQUEST.md \
  main.py requirements.txt diagnostic_tool.py pytest.ini \
  src/ scripts/ tests/

git commit -m "docs: nominal native SR pipeline; comments; stacked residual diagnostic

- Document default main.py overrides and outputs (docs/NOMINAL_NATIVE_SCIENCE_RUN.md)
- README usage + link to doc; ignore .venv/.cursor/.local_env
- Comments on nominal_overrides, pipeline native crop/unmask, GP opt guard
- Expand docstrings for crop/unmask and stacked transient residual PDF"

git push -u origin feature/nominal-native-science-run-docs
```

**Open a PR:** after `git push`, GitHub prints a “Create pull request” URL for the new branch, or use  
`https://github.com/<org>/<repo>/compare/main...<your-branch>`.

With [GitHub CLI](https://cli.github.com/) installed and authenticated:

```bash
gh pr create --base main --title "Nominal native SR run: docs, comments, PR prep" --body "$(head -n 40 docs/NOMINAL_NATIVE_SCIENCE_RUN.md)"
```

Adjust the branch name and PR body as needed. Do not commit `data/`, `output/`, `.venv/`, or large scratch trees (see `.gitignore`).

## Approve and merge (example: PR #9)

For [PR #9](https://github.com/astrofoley/spitzer_photometry/pull/9) (“feat: native SR pipeline docs, comments, integrated science defaults”), GitHub reported **mergeable** / **clean** with respect to `main` at creation time. To ship:

1. Review the **Files changed** tab.
2. **Approve** if your workflow requires it, then **Merge pull request** (choose merge or squash to taste).

## After merge — update your local clone

```bash
git checkout main
git pull origin main
git branch -d feature/nominal-native-science-docs-pr   # safe after merge; omit if you keep the branch
```

Optional: delete the remote feature branch from GitHub (**Branches** page or “Delete branch” on the merged PR).

## Verification before merge

Quick syntax check:

```bash
python -m py_compile main.py src/pipeline_fit.py src/config.py
```

Full tests (`pytest tests/`) can be slow; run when you have time and data/PRF layout available.
