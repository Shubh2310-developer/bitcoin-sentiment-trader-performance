# Phase 10: Documentation & Architecture
## Repository: bitcoin-sentiment-trader-performance
### Engineering Standards

---

## 1. Objective

Populate and finalize all project documentation: `docs/architecture.md`, `docs/methodology.md`, `docs/data_dictionary.md`, `docs/index.md`, and `docs/api_reference/`. Generate API documentation from code docstrings. Ensure README.md is current with quickstart and directory overview. Run the documentation audit command.

**Note:** Documentation is maintained incrementally throughout all phases. This phase is a final completeness and quality gate before release.

---

## 2. Prerequisites

- [ ] Phase 09 complete (Go/No-Go gate cleared)
- [ ] All `src/` modules have Google-style docstrings on every public function and class
- [ ] `configs/logging.yaml` configured
- [ ] API doc generation tool configured (pdoc, Sphinx, or mkdocs with autodoc)

---

## 3. Pipeline Stage Alignment

Documentation standards apply throughout the entire project lifecycle, not only in this phase. This phase enforces completeness before release.

---

## 4. Agent Assignment

| Role | Agent | Activation |
|---|---|---|
| **Primary** | `technical-writer` | `/agent:technical-writer` |
| **Supporting** | `code-reviewer` | `/agent:code-reviewer` |

---

## 5. Skill Invocations

| Skill | Activation | Purpose |
|---|---|---|
| `code-reviewer` | `/skill:code-reviewer` | Documentation completeness audit, docstring quality review |

---

## 6. Command Invocations

| Command | Activation | Purpose |
|---|---|---|
| `docs-maintenance` | `/docs-maintenance` | Full documentation audit: find missing docstrings, stale docs, broken links |
| `generate-api-documentation` | `/generate-api-documentation` | Auto-generate `docs/api_reference/` from source docstrings |
| `update-docs` | `/update-docs` | Final sync of methodology and data dictionary with pipeline changes |

---

## 7. Documentation Completeness Checklist

### 7.1 `docs/index.md`
- [ ] Project purpose stated
- [ ] Quickstart (env setup + run full pipeline)
- [ ] Links to all phase documents
- [ ] Link to the engineering standards document
- [ ] Table of contents for all docs

### 7.2 `docs/architecture.md`
- [ ] System architecture diagram (ASCII or mermaid)
- [ ] Module catalog with responsibilities
- [ ] Data flow diagram
- [ ] CI/CD architecture section
- [ ] ADR log up to date

### 7.3 `docs/methodology.md`
- [ ] All null handling decisions documented (§2.4 table complete)
- [ ] All statistical test selection decisions documented (§4.1–§4.7)
- [ ] ML methodology documented if Phase 08 invoked
- [ ] Amendment log current

### 7.4 `docs/data_dictionary.md`
- [ ] All raw dataset columns documented
- [ ] All processed dataset columns documented
- [ ] All engineered features documented with formula, window, look-ahead status
- [ ] Known limitations table populated

### 7.5 `README.md`
- [ ] Project purpose (2–3 sentences)
- [ ] Quickstart: `conda env create`, activate, run full pipeline, run tests
- [ ] Directory overview (top-level structure explained)
- [ ] Link to the engineering standards document
- [ ] Link to `docs/index.md`

### 7.6 `docs/api_reference/`
- [ ] Auto-generated from docstrings via `generate-api-documentation` command
- [ ] Covers all public functions in `src/sentiment_trader_analytics/`
- [ ] No hand-written API docs that could drift from code

---

## 8. Docstring Coverage Audit

Every public function, class, and module in `src/` must have a Google-style docstring with:
- Purpose (one sentence)
- `Args:` section with type and description for every parameter
- `Returns:` section with type and description
- `Raises:` section for any explicitly raised exceptions
- Statistical assumptions documented where relevant

**Audit command:**
```bash
# Check docstring coverage
python -m interrogate src/sentiment_trader_analytics/ \
    --verbose \
    --fail-under 100 \
    --ignore-init-method \
    --ignore-magic
```

---

## 9. Verification Commands

```bash
# Run documentation audit command
# /docs-maintenance

# Generate API documentation
# /generate-api-documentation

# Check all docs files are non-empty
for f in docs/index.md docs/architecture.md docs/methodology.md docs/data_dictionary.md; do
    wc -l "$f"
done

# Verify README quickstart works
cat README.md | grep -A 5 "Quickstart"

# Check docstring coverage
python -m interrogate src/ --fail-under 100

# Verify API reference generated
ls docs/api_reference/
```

---

## 10. Go / No-Go Gate

| Check | Verification |
|---|---|
| All docs files non-empty and current | `wc -l docs/*.md` all > 50 lines |
| API reference generated | Files present in `docs/api_reference/` |
| 100% docstring coverage on public functions | `interrogate` exits 0 |
| README quickstart current | Contains conda and pytest commands |
| `docs/methodology.md` amendment log current | No undocumented analytical decisions |
| Documentation audit passes | `/docs-maintenance` command returns no critical issues |

---

*Governed by §20. Proceed to [Phase 11](phase_11_testing_release.md) upon gate clearance.*
