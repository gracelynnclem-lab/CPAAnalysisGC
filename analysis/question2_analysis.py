#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

LIKERT_Q52_ORDER = [
    "Significantly decreased desire",
    "Decreased desire",
    "No change in desire",
    "Increased desire",
    "Significantly increased desire",
]

LIKELIHOOD_Q35_ORDER = [
    "Extremely unlikely",
    "Somewhat unlikely",
    "Neither likely nor unlikely",
    "Somewhat likely",
    "Extremely likely",
]


def to_snake_case(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def read_and_clean(csv_path: Path) -> list[dict[str, str | None]]:
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    cleaned = []
    for row in raw_rows:
        if not row.get("ResponseId", "").startswith("R_"):
            continue
        if row.get("Finished") not in {"True", "False"}:
            continue

        parsed = {to_snake_case(k): (v.strip() or None) for k, v in row.items()}
        cleaned.append(
            {
                "responseid": parsed.get("responseid"),
                "q27": parsed.get("q27"),
                "q52": parsed.get("q52"),
                "q53": parsed.get("q53"),
                "q31": parsed.get("q31"),
                "q35": parsed.get("q35"),
                "q58": parsed.get("q58"),
            }
        )
    return cleaned


def derive_fields(rows: list[dict[str, str | None]]) -> list[dict[str, str | int | float | None]]:
    out = []
    for r in rows:
        student_level = r["q27"] if r["q27"] in {"Undergraduate", "Graduate"} else None
        aware = r["q53"] if student_level == "Undergraduate" else r["q31"]

        if student_level == "Graduate":
            enroll = "Currently enrolled (graduate respondent)"
        elif student_level == "Undergraduate":
            enroll = "Planned enrollment status not explicitly asked"
        else:
            enroll = "Missing"

        reduced = None
        if student_level == "Undergraduate" and r["q52"] in LIKERT_Q52_ORDER:
            reduced = 1 if r["q52"] in {"Decreased desire", "Significantly decreased desire"} else 0
        elif student_level == "Graduate" and r["q35"] in LIKELIHOOD_Q35_ORDER:
            reduced = 1 if r["q35"] in {"Extremely unlikely", "Somewhat unlikely"} else 0

        row = dict(r)
        row["student_level"] = student_level
        row["aware_alternative_pathway"] = aware
        row["current_or_planned_grad_enrollment"] = enroll
        row["reduced_grad_degree_interest"] = reduced
        out.append(row)
    return out


def group_counts(rows, keys):
    counts = {}
    for r in rows:
        k = tuple(r.get(key) for key in keys)
        counts[k] = counts.get(k, 0) + 1
    result = []
    for k, v in sorted(counts.items(), key=lambda x: tuple("" if t is None else str(t) for t in x[0])):
        obj = {key: k[i] for i, key in enumerate(keys)}
        obj["respondent_count"] = v
        result.append(obj)
    return result


def outcome_rates(rows):
    buckets: dict[tuple[str | None, str | None], list[int]] = {}
    for r in rows:
        y = r.get("reduced_grad_degree_interest")
        if y is None:
            continue
        key = (r.get("student_level"), r.get("aware_alternative_pathway"))
        buckets.setdefault(key, []).append(int(y))

    out = []
    for (level, aware), ys in sorted(buckets.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
        n = len(ys)
        out.append(
            {
                "student_level": level,
                "aware_alternative_pathway": aware,
                "n": n,
                "reduced_interest_rate": sum(ys) / n,
            }
        )
    return out



def safe_exp(x: float) -> float:
    return math.exp(max(min(x, 700), -700))

def sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1 / (1 + ez)
    ez = math.exp(z)
    return ez / (1 + ez)


def mat_inv(a):
    n = len(a)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(a)]
    for i in range(n):
        pivot = max(range(i, n), key=lambda r: abs(aug[r][i]))
        if abs(aug[pivot][i]) < 1e-12:
            raise RuntimeError("Matrix inversion failed: singular matrix")
        aug[i], aug[pivot] = aug[pivot], aug[i]
        factor = aug[i][i]
        aug[i] = [x / factor for x in aug[i]]
        for r in range(n):
            if r == i:
                continue
            f = aug[r][i]
            aug[r] = [aug[r][c] - f * aug[i][c] for c in range(2 * n)]
    return [row[n:] for row in aug]


def logistic_regression(rows):
    model_rows = [
        r
        for r in rows
        if r.get("reduced_grad_degree_interest") is not None
        and r.get("aware_alternative_pathway") in {"Yes", "No"}
        and r.get("student_level") in {"Undergraduate", "Graduate"}
    ]
    x = []
    y = []
    for r in model_rows:
        aware = 1.0 if r["aware_alternative_pathway"] == "Yes" else 0.0
        grad = 1.0 if r["student_level"] == "Graduate" else 0.0
        x.append([1.0, aware, grad])
        y.append(float(r["reduced_grad_degree_interest"]))

    p = 3
    beta = [0.0] * p
    for _ in range(50):
        probs = [sigmoid(sum(beta[j] * xi[j] for j in range(p))) for xi in x]
        grad = [0.0] * p
        hess = [[0.0] * p for _ in range(p)]
        for i, xi in enumerate(x):
            w = probs[i] * (1 - probs[i])
            for a in range(p):
                grad[a] += xi[a] * (y[i] - probs[i])
                for b in range(p):
                    hess[a][b] += xi[a] * xi[b] * w
        for d in range(p):
            hess[d][d] += 1e-6
        inv_h = mat_inv(hess)
        step = [sum(inv_h[r][c] * grad[c] for c in range(p)) for r in range(p)]
        beta_new = [beta[j] + step[j] for j in range(p)]
        if max(abs(beta_new[j] - beta[j]) for j in range(p)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new

    probs = [sigmoid(sum(beta[j] * xi[j] for j in range(p))) for xi in x]
    hess = [[0.0] * p for _ in range(p)]
    for i, xi in enumerate(x):
        w = probs[i] * (1 - probs[i])
        for a in range(p):
            for b in range(p):
                hess[a][b] += xi[a] * xi[b] * w
    for d in range(p):
        hess[d][d] += 1e-6
    cov = mat_inv(hess)
    se = [math.sqrt(max(cov[i][i], 0.0)) for i in range(p)]

    terms = ["const", "aware_yes", "graduate"]
    out = []
    for i, t in enumerate(terms):
        z = beta[i] / se[i] if se[i] > 0 else 0.0
        pval = math.erfc(abs(z) / math.sqrt(2))
        ci_low = beta[i] - 1.96 * se[i]
        ci_high = beta[i] + 1.96 * se[i]
        out.append(
            {
                "term": t,
                "coef_log_odds": beta[i],
                "odds_ratio": safe_exp(beta[i]),
                "ci_lower_odds_ratio": safe_exp(ci_low),
                "ci_upper_odds_ratio": safe_exp(ci_high),
                "p_value": pval,
            }
        )

    return out, len(model_rows)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_svg_regression(path: Path, regression_rows: list[dict]):
    rows = [r for r in regression_rows if r["term"] != "const"]
    labels = {
        "aware_yes": "Aware of alternative pathway",
        "graduate": "Graduate respondent",
        "interaction": "Awareness × Graduate",
    }
    width, height = 900, 360
    left, right = 280, 70
    top, bottom = 40, 50
    plot_w = width - left - right

    # log-scale range.
    all_vals = [
        max(0.05, float(r["ci_lower_odds_ratio"])) for r in rows
    ] + [min(20.0, float(r["ci_upper_odds_ratio"])) for r in rows] + [1.0]
    lo, hi = min(all_vals), max(all_vals)
    lo, hi = math.log10(lo), math.log10(hi)

    def x_pos(v):
        v = max(0.05, min(20.0, float(v)))
        return left + (math.log10(v) - lo) / (hi - lo) * plot_w

    y_positions = [top + i * 95 for i in range(len(rows))]
    elements = []
    elements.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')

    x1 = x_pos(1.0)
    elements.append(f'<line x1="{x1:.1f}" y1="{top-10}" x2="{x1:.1f}" y2="{height-bottom+10}" stroke="#888" stroke-dasharray="5,4"/>')

    for y, r in zip(y_positions, rows):
        xl = x_pos(r["ci_lower_odds_ratio"])
        xm = x_pos(r["odds_ratio"])
        xu = x_pos(r["ci_upper_odds_ratio"])
        label = labels.get(r["term"], r["term"])
        elements.append(f'<line x1="{xl:.1f}" y1="{y}" x2="{xu:.1f}" y2="{y}" stroke="#2b6cb0" stroke-width="3"/>')
        elements.append(f'<circle cx="{xm:.1f}" cy="{y}" r="6" fill="#2c5282"/>')
        elements.append(f'<text x="20" y="{y+5}" font-size="15" font-family="Arial">{label}</text>')
        elements.append(f'<text x="{xu+10:.1f}" y="{y+5}" font-size="12" font-family="Arial" fill="#333">OR={r["odds_ratio"]:.2f}</text>')

    ticks = [0.1, 0.25, 0.5, 1, 2, 4, 8, 16]
    for t in ticks:
        if t < 10 ** lo or t > 10 ** hi:
            continue
        xt = x_pos(t)
        elements.append(f'<line x1="{xt:.1f}" y1="{height-bottom}" x2="{xt:.1f}" y2="{height-bottom+6}" stroke="#333"/>')
        elements.append(f'<text x="{xt-10:.1f}" y="{height-bottom+24}" font-size="12" font-family="Arial">{t:g}</text>')

    elements.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>')
    elements.append(f'<text x="{left+plot_w/2-100:.1f}" y="{height-10}" font-size="14" font-family="Arial">Odds ratio (log scale)</text>')
    elements.append('<text x="20" y="24" font-size="18" font-family="Arial" font-weight="bold">Association with reduced graduate-degree interest</text>')

    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">{"".join(elements)}</svg>'
    path.write_text(svg, encoding="utf-8")


def write_markdown(path: Path, n_total: int, counts, outcomes, regression, model_n: int):
    lines = [
        "# Question 2 Analysis Summary",
        "",
        "## Research question",
        "Does awareness of the alternative CPA pathway correspond to reduced interest in pursuing a graduate accounting degree?",
        "",
        "## Data preparation",
        f"- Retained {n_total} valid survey responses after removing Qualtrics metadata rows.",
        "- Treated Likert responses as ordered categorical scales (`Q52` and `Q35`).",
        "- Explicitly coded unavailable planned-enrollment responses for undergraduates.",
        "",
        "## Segment counts (awareness × student level)",
        "| student_level | aware_alternative_pathway | respondent_count |",
        "|---|---:|---:|",
    ]
    for r in counts:
        lines.append(f"| {r['student_level']} | {r['aware_alternative_pathway']} | {r['respondent_count']} |")

    lines.extend([
        "",
        "## Reduced-interest rates by segment",
        "| student_level | aware_alternative_pathway | n | reduced_interest_rate |",
        "|---|---:|---:|---:|",
    ])
    for r in outcomes:
        lines.append(f"| {r['student_level']} | {r['aware_alternative_pathway']} | {r['n']} | {100*r['reduced_interest_rate']:.1f}% |")

    lines.extend([
        "",
        "## Logistic regression (descriptive association only)",
        "Outcome: `reduced_grad_degree_interest` (1 = reduced inclination toward graduate degree).",
        "",
        "| term | odds_ratio | ci_lower_odds_ratio | ci_upper_odds_ratio | p_value |",
        "|---|---:|---:|---:|---:|",
    ])
    for r in regression:
        lines.append(
            f"| {r['term']} | {r['odds_ratio']:.3f} | {r['ci_lower_odds_ratio']:.3f} | {r['ci_upper_odds_ratio']:.3f} | {r['p_value']:.4f} |"
        )

    lines.extend([
        "",
        f"Model sample size: {model_n} respondents.",
        "",
        "**Interpretation note:** This analysis is descriptive and does not support causal claims.",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


def validate(paths: list[Path]):
    missing = [str(p) for p in paths if not p.exists() or p.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Missing or empty outputs: {missing}")


def main():
    repo = Path(__file__).resolve().parents[1]
    input_csv = repo / "Alternative CPA Pathways Survey_December 31, 2025_09.45.csv"
    out_dir = repo / "outputs"
    out_dir.mkdir(exist_ok=True)

    rows = derive_fields(read_and_clean(input_csv))
    counts = group_counts(rows, ["student_level", "aware_alternative_pathway"])
    outcomes = outcome_rates(rows)
    regression, model_n = logistic_regression(rows)

    cleaned_csv = out_dir / "question2_cleaned_dataset.csv"
    counts_csv = out_dir / "question2_segment_counts.csv"
    outcomes_csv = out_dir / "question2_outcome_rates.csv"
    reg_csv = out_dir / "question2_logistic_regression.csv"
    reg_json = out_dir / "question2_logistic_regression.json"
    summary_md = out_dir / "question2_summary.md"
    plot_svg = out_dir / "question2_logistic_regression_plot.svg"

    write_csv(cleaned_csv, rows)
    write_csv(counts_csv, counts)
    write_csv(outcomes_csv, outcomes)
    write_csv(reg_csv, regression)
    reg_json.write_text(json.dumps(regression, indent=2), encoding="utf-8")
    write_markdown(summary_md, len(rows), counts, outcomes, regression, model_n)
    write_svg_regression(plot_svg, regression)

    validate([cleaned_csv, counts_csv, outcomes_csv, reg_csv, reg_json, summary_md, plot_svg])


if __name__ == "__main__":
    main()
