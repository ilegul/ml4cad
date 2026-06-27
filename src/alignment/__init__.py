"""
src/alignment/
──────────────
Modules to *retrace the methodological steps* of

    Pingitore et al., "Machine learning to identify a composite indicator to
    predict cardiac death in ischemic heart disease",
    International Journal of Cardiology 404 (2024) 131981.

on the project's **strict** cohort, so that the thesis results (extension with
thyroid function) become comparable with that paper.

Goal: NOT to reproduce the same numbers, but to retrace the same steps while
being explicit about the differences (no creatinine/eGFR, 5-fold CV added next
to the 60/20/20 split, 5-dummy thyroid-state encoding, …).

Everything reuses the existing cohort construction and feature engineering from
``src`` / ``configs`` — nothing here modifies datasets, cohorts, existing
modules or reports. These modules only *add*.
"""
