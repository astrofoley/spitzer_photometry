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
