"""
esegui_esperimento.py — runs one step of the pipeline with CONFIG overrides
from the command line, so the same scripts can be run for different heatmap
modes without editing them. Every output (checkpoints, calibration) goes to
--dir. Used by run_tutto.slurm.

    python esegui_esperimento.py heatmap  --mode fixed --dir esperimenti/fixed
    python esegui_esperimento.py refine   --dir esperimenti/fixed
    python esegui_esperimento.py evaluate --dir esperimenti/fixed
    python esegui_esperimento.py refine   --dir esperimenti_s2/B --variante B

The evaluate step combines the two stages with the fusion parameters of
<dir>/parametri_stage2.py when that file exists (a Stage 2 variant needs its
own, re-fitted with strategie_stage2.py --salva), otherwise with
src/parametri_stage2.py.

Run each step in its own process: train_refine / evaluate_refine read the
Stage 2 mask from src/landmark_stats.py at import time, and that file is
regenerated at the end of the heatmap step.
"""
import argparse
import importlib.util
import json
import os


def parametri_fusione(cartella):
    """Fusion parameters of <cartella>/parametri_stage2.py, or None."""
    path = os.path.join(cartella, "parametri_stage2.py")
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location("parametri_stage2_esperimento", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return path, mod


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["heatmap", "refine", "evaluate"])
    ap.add_argument("--dir", required=True, help="experiment folder (checkpoints, calibration)")
    ap.add_argument("--mode", choices=["fixed", "learned_sigma", "probabilistic"],
                    help="heatmap_mode (heatmap step only)")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--tracciati", default=None,
                    help="Tracciati folder (default: not used, as in the previous runs)")
    ap.add_argument("--tabella-scale", default=None, help="per-case scale table for Tracciati")
    ap.add_argument("--no-tta", action="store_true", help="evaluate without TTA")
    ap.add_argument("--insieme", default="validation",
                    choices=["validation", "test", "test_aariz", "test_isbi1", "test_isbi2", "unife"],
                    help="evaluate step: images to evaluate on (test sets: official split only)")
    ap.add_argument("--variante", choices=["base", "A", "B"],
                    help="refine step: Stage 2 variant (see VARIANTI in train_refine.py; default: CONFIG)")
    a = ap.parse_args(argv)

    os.makedirs(a.dir, exist_ok=True)
    stage1 = os.path.join(a.dir, "modello_heatmap.pth")
    stage2 = os.path.join(a.dir, "modello_refine.pth")

    if a.step == "heatmap":
        import train_heatmap as m
        m.CONFIG.update(save_path=stage1, tracciati_path=a.tracciati,
                        tracciati_tabella_scale=a.tabella_scale)
        if a.mode:
            m.CONFIG["heatmap_mode"] = a.mode
    elif a.step == "refine":
        import train_refine as m
        m.CONFIG.update(stage1_path=stage1, save_path=stage2, tracciati_path=a.tracciati,
                        tracciati_tabella_scale=a.tabella_scale)
        if a.variante:
            m.applica_variante(m.CONFIG, a.variante)
    else:
        import evaluate_refine as m
        m.CONFIG.update(stage1_path=stage1, stage2_path=stage2,
                        calibrazione_path=os.path.join(a.dir, "calibrazione_incertezza.json"))
        if a.no_tta:
            m.CONFIG["usa_tta"] = False
        m.CONFIG["insieme"] = a.insieme
        par = parametri_fusione(a.dir)
        if par:
            path, mod = par
            m.CONFIG.update(selezione_stage2=mod.SELEZIONE_STAGE2, fusione_stage2=dict(mod.FUSIONE_STAGE2),
                            soglia_sigma_tta_mm=mod.SOGLIA_SIGMA_TTA_MM)
            print(f"[esegui_esperimento] fusion parameters from {path}: {mod.FUSIONE_STAGE2}")

    if a.batch_size:
        m.CONFIG["batch_size"] = a.batch_size
    if a.epochs and "num_epochs" in m.CONFIG:
        m.CONFIG["num_epochs"] = a.epochs

    print(f"[esegui_esperimento] step={a.step} dir={a.dir}")
    print(json.dumps({k: v for k, v in m.CONFIG.items()}, indent=1, default=str))
    m.main()


if __name__ == "__main__":
    main()
