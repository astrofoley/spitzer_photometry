# Pull request: Phase 1 publication baseline (tooling + WCS)

Use this as the **PR title** and **description** on GitHub. Replace issue numbers after you match each item to an existing issue (see [`docs/github_issues/ISSUES_FOR_GITHUB.md`](../github_issues/ISSUES_FOR_GITHUB.md)).

---

## Title (copy)

```
feat: Phase 1 publication baseline — channel clarity, WCS warnings, run manifest, pytest CI
```

---

## Description (copy)

### Summary

Implements **[Phase 1](https://github.com/astrofoley/spitzer_photometry/blob/main/docs/PUBLICATION_FIX_ORDER.md)** (*Baseline honesty and tooling*): single-band clarity, reduced Astropy WCS warning noise, automatic **`run_manifest.json`**, fast pytest subset + GitHub Actions, plus SIP-aware projection cache keys and tests.

### Closes (edit `#NN` to match your repo’s issue numbers)

Replace each line with the real GitHub issue number, or delete lines that do not have an issue yet.

```
Closes #<!-- BUG-3 issue number -->
Closes #<!-- BUG-1 issue number -->
Closes #<!-- FR-7 issue number -->
Closes #<!-- BUG-2 issue number -->
```

**Mapping (from internal labels → [`ISSUES_FOR_GITHUB.md`](../github_issues/ISSUES_FOR_GITHUB.md)):**

| Label | Typical title contains |
|-------|-------------------------|
| **BUG-3** | Pipeline / one IRAC band / `CHANNEL` |
| **BUG-1** | `cdelt` ignored / WCS warning spam |
| **FR-7** | run manifest / reproducibility |
| **BUG-2** | pytest slow / markers / CI |

Optional related (no closing keyword unless you filed separate issues):

- **BUG-1b / projection cache:** SIP fingerprint for `_PROJECTION_CACHE` — same PR as BUG-1 or split into its own issue.

### What changed

- **BUG-3:** `chan_str` from `config.CHANNEL`; channel filter logging in `find_spitzer_files`; README; config keys for tests (`FLOAT_NUCLEAR_POINT_POSITION`, `NUCLEAR_POINT_POS_RIDGE`, `GP_COMPONENTS_NONNEGATIVE`).
- **BUG-1:** Filter redundant CD/CDELT warnings in `solver.py` + `pytest.ini`; SIP distortion fingerprint for projection operator cache; tests + optional `scripts/verify_cbcd_sip_headers.py`.
- **FR-7:** `src/run_manifest.py`; `pipeline_fit` writes `OUTPUT_DIR/run_manifest.json` after successful fit; `tests/test_run_manifest.py`.
- **BUG-2:** `@pytest.mark.slow` on heavy tests; README (`pytest -m "not slow"`); `.github/workflows/pytest.yml`; supporting fixes (`GP_KERNEL_TYPE` diagonal mode, phase0 test strings, skips).
- **Docs:** `docs/PUBLICATION_FIX_ORDER.md` Phase 1 completion table; `docs/PUBLICATION_READINESS.md` native WCS vs linear scene paragraph.

### How to test

```bash
pip install -r requirements.txt
pytest -m "not slow" --tb=short
```

Full suite (includes slow):

```bash
pytest --tb=short
```

---

## After you push

```bash
git push -u origin <branch-name>
```

Then open a PR from that branch into `main` and paste the title/description above. In the GitHub PR sidebar, **Development** → link issues, or ensure `Closes #NN` lines use correct numbers.
