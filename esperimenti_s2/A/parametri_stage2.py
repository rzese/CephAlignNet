"""
How Stage 2 is combined with Stage 1 — fixed parameters, read by
evaluate_refine.py and predict_heatmap.py.

Written by strategie_stage2.py on 2026-10-30 from esperimenti_s2/A/analisi_stage2.csv
(parameters re-fitted on all the images; table = 2-fold CV over even/odd images,
objective = mean accuracy within 2 / 2.5 / 3 / 4 mm):

  strategy                              MRE    <2mm  <2.5mm    <3mm    <4mm mean acc
  Stage 1 only                        1.069   86.8%   92.2%   94.8%   97.4%   92.80%
  Stage 2 everywhere                  1.389   81.2%   86.5%   90.3%   94.0%   88.01%
  threshold on sigma_tta              1.068   86.9%   92.3%   94.9%   97.4%   92.88%
  fusion (weighted average)           1.017   88.5%   92.8%   95.0%   97.6%   93.47%
  classifier                          1.037   87.6%   92.5%   94.9%   97.5%   93.14%

Not re-tuned automatically after a new training. To measure again:
    python analisi_stage2.py --dir <experiment>
    python strategie_stage2.py <experiment>/analisi_stage2.csv --salva src/parametri_stage2.py
"""

# "fusione" | "incertezza" | "tutti" | "maschera"  (see evaluate_refine.py)
SELEZIONE_STAGE2 = "fusione"

# Fusion weight of Stage 2: w = s1^p / (s1^p + (c * s2)^p), s1 = sqrt(sigma_tta^2 + floor^2)
FUSIONE_STAGE2 = {"c": 0.4, "p": 2, "floor_mm": 0.2}

# Used only with SELEZIONE_STAGE2 = "incertezza": Stage 2 where sigma_tta > threshold
SOGLIA_SIGMA_TTA_MM = 1.19
