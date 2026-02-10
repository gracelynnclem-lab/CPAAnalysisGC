# CPAAnalysisGC

## Question 2 Reproducible Analysis

### What Question 2 evaluates
Question 2 asks: **Does awareness of the alternative CPA licensure pathway correspond to reduced perceived need for a graduate accounting degree?**

This repository includes an automated, descriptive (non-causal) analysis that compares outcomes by:
- awareness vs non-awareness of the alternative pathway,
- undergraduate vs graduate respondent status.

### What the script produces
Run:

```bash
python analysis/question2_analysis.py
```

The script reads the Qualtrics CSV, removes metadata rows, standardizes fields, and writes:
- `outputs/question2_cleaned_dataset.csv`
- `outputs/question2_segment_counts.csv`
- `outputs/question2_outcome_rates.csv`
- `outputs/question2_logistic_regression.csv`
- `outputs/question2_logistic_regression.json`
- `outputs/question2_summary.md`
- `outputs/question2_logistic_regression_plot.svg`

### How GitHub Actions enforces reproducibility
The workflow at `.github/workflows/question2-analysis.yml` runs on **every pull request to `main`**. It:
1. checks out the repository,
2. sets up Python,
3. installs pinned dependencies from `analysis/requirements.txt`,
4. executes the analysis script,
5. validates required output files,
6. uploads outputs as workflow artifacts.

If the script errors or any required output is missing/empty, the workflow fails.
