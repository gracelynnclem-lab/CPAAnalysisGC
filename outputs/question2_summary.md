# Question 2 Analysis Summary

## Research question
Does awareness of the alternative CPA pathway correspond to reduced interest in pursuing a graduate accounting degree?

## Data preparation
- Retained 206 valid survey responses after removing Qualtrics metadata rows.
- Treated Likert responses as ordered categorical scales (`Q52` and `Q35`).
- Explicitly coded unavailable planned-enrollment responses for undergraduates.

## Segment counts (awareness × student level)
| student_level | aware_alternative_pathway | respondent_count |
|---|---:|---:|
| None | None | 4 |
| Graduate | No | 40 |
| Graduate | Yes | 15 |
| Undergraduate | No | 58 |
| Undergraduate | Yes | 89 |

## Reduced-interest rates by segment
| student_level | aware_alternative_pathway | n | reduced_interest_rate |
|---|---:|---:|---:|
| Graduate | No | 40 | 20.0% |
| Undergraduate | No | 58 | 6.9% |
| Undergraduate | Yes | 89 | 48.3% |

## Logistic regression (descriptive association only)
Outcome: `reduced_grad_degree_interest` (1 = reduced inclination toward graduate degree).

| term | odds_ratio | ci_lower_odds_ratio | ci_upper_odds_ratio | p_value |
|---|---:|---:|---:|---:|
| const | 0.074 | 0.027 | 0.205 | 0.0000 |
| aware_yes | 12.620 | 4.211 | 37.815 | 0.0000 |
| graduate | 3.375 | 0.941 | 12.107 | 0.0620 |

Model sample size: 187 respondents.

**Interpretation note:** This analysis is descriptive and does not support causal claims.