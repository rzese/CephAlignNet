"""
How Stage 2 is combined with Stage 1 — fixed parameters, read by
evaluate_refine.py and predict_heatmap.py.

Measured on the FIRST run (Stage 1 "probabilistic" + image-based Stage 2,
esperimenti/probabilistic) with analisi_stage2.py + strategie_stage2.py:
2-fold cross-validation over the even/odd validation images, objective =
mean accuracy within 2 / 2.5 / 3 / 4 mm.

  strategy                    MRE    <2mm  <2.5mm   <3mm   <4mm  mean acc
  Stage 1 only               1.085  86.6%  91.4%  94.3%  97.3%  92.41%
  threshold on sigma_tta     1.063  87.3%  91.9%  94.5%  97.4%  92.80%
  fusion (weighted average)  1.029  88.4%  92.5%  94.8%  97.5%  93.30%

These values are NOT re-tuned automatically after a new training (as for
src/landmark_stats.py, the first run is the empirical measurement and the
following runs use it). To measure them again on a new run:
    python analisi_stage2.py --dir <experiment>
    python strategie_stage2.py <experiment>/analisi_stage2.csv --salva src/parametri_stage2.py
"""

# "fusione" | "incertezza" | "tutti" | "maschera"  (see evaluate_refine.py)
SELEZIONE_STAGE2 = "fusione"

# Fusion weight of Stage 2: w = s1^p / (s1^p + (c * s2)^p), s1 = sqrt(sigma_tta^2 + floor^2)
FUSIONE_STAGE2 = {"c": 0.32, "p": 2.0, "floor_mm": 0.0}

# Used only with SELEZIONE_STAGE2 = "incertezza": Stage 2 where sigma_tta > threshold
SOGLIA_SIGMA_TTA_MM = 0.25
