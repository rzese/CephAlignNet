"""
check_coords.py
===============
Verifica rapida PRE-TRAINING: controlla che le coordinate di tutti
i dataset siano correttamente normalizzate in [0,1] e che i Tracciati
non avvelenino il training.

Esegui dalla root del progetto:
    python3 check_coords.py

Output atteso:
    [OK] AARIZ: N campioni, coordinate in [0.xxx, 0.xxx]
    [OK] ISBI: N campioni, coordinate in [0.xxx, 0.xxx]
    [OK] TRACCIATI: N campioni, coordinate in [0.xxx, 0.xxx]
    [OK] Pool totale: N campioni
"""

import sys, os
sys.path.insert(0, "src")

import torch
from dataset           import AarizDataset
from dataset_isbi      import ISBIDataset
from dataset_tracciati import TracciatiDataset
from datasets          import Datasets

AARIZ_PATH     = "data/Aariz"
ISBI_PATH      = "data/Cepha400"
TRACCIATI_PATH = "data/tracciati_forniti"
N_CHECK        = 30   # campioni da verificare per dataset


def check_dataset(nome, ds, n_check=N_CHECK):
    n_check = min(n_check, len(ds))
    errori  = []
    n_validi_tot = 0

    for i in range(n_check):
        sample = ds[i]
        pts = sample[1]

        # Individua landmark non-zero (= annotati)
        mask_nonzero = (pts[:, 0] != 0.0) | (pts[:, 1] != 0.0)
        validi = pts[mask_nonzero]

        if len(validi) == 0:
            errori.append(f"  campione {i}: NESSUN landmark valido")
            continue

        n_validi_tot += len(validi)
        mn, mx = validi.min().item(), validi.max().item()

        if mn < -0.001:
            errori.append(f"  campione {i}: coordinate negative (min={mn:.4f})")
        if mx > 1.001:
            errori.append(f"  campione {i}: coordinate > 1 (max={mx:.4f}) — "
                          f"probabilmente NON normalizzate (pixel grezzi?)")
        if mx < 0.01:
            errori.append(f"  campione {i}: coordinate troppo piccole (max={mx:.5f}) — "
                          f"coord_scale probabilmente sbagliato")

    # Stampa riepilogo
    sample0 = ds[0]
    pts0 = sample0[1]
    validi0 = pts0[(pts0[:, 0] != 0.0) | (pts0[:, 1] != 0.0)]
    rng = f"[{validi0.min():.4f}, {validi0.max():.4f}]" if len(validi0) > 0 else "N/A"

    if errori:
        print(f"[FAIL] {nome}: {len(errori)}/{n_check} campioni con errori")
        for e in errori[:5]:
            print(e)
        return False
    else:
        print(f"[OK]   {nome}: {n_check} campioni OK | "
              f"range coord primo campione: {rng} | "
              f"landmark medi per campione: {n_validi_tot/n_check:.1f}")
        return True


print("=" * 60)
print("CHECK COORDINATE PRE-TRAINING")
print("=" * 60)

tutti_ok = True

# ── AARIZ ──────────────────────────────────────────────────────────────────
try:
    aariz = AarizDataset(base_path=AARIZ_PATH, split='all')
    tutti_ok &= check_dataset("AARIZ", aariz)
except Exception as e:
    print(f"[FAIL] AARIZ: {e}")
    tutti_ok = False

# ── ISBI ───────────────────────────────────────────────────────────────────
try:
    isbi = ISBIDataset(base_path=ISBI_PATH, split='all')
    tutti_ok &= check_dataset("ISBI", isbi)
except Exception as e:
    print(f"[FAIL] ISBI: {e}")
    tutti_ok = False

# ── TRACCIATI ──────────────────────────────────────────────────────────────
if os.path.isdir(TRACCIATI_PATH):
    try:
        tracciati = TracciatiDataset(
            base_path=TRACCIATI_PATH,
            subfolder_names=["tracciato 50", "tracciato 50_2"],
            coord_scale=0.01,
            flip_y=False,
            use_thumbnail=False,
        )
        if len(tracciati) > 0:
            tutti_ok &= check_dataset("TRACCIATI", tracciati)
            # Verifica extra: mostra i valori reali del primo campione
            img, pts = tracciati[0]
            validi = pts[(pts[:, 0] != 0.0) | (pts[:, 1] != 0.0)]
            print(f"       Primo campione Tracciati: {tracciati.samples[0][0]}")
            print(f"       Coord min={validi.min():.4f} max={validi.max():.4f} "
                  f"mean={validi.mean():.4f}")
        else:
            print(f"[WARN] TRACCIATI: 0 campioni trovati in {TRACCIATI_PATH}")
    except Exception as e:
        print(f"[FAIL] TRACCIATI: {e}")
        tutti_ok = False
else:
    print(f"[WARN] TRACCIATI: path non trovato ({TRACCIATI_PATH}) — verrà ignorato")

# ── POOL AGGREGATO ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
try:
    ds_total = Datasets(
        aariz_path=AARIZ_PATH,
        isbi_path=ISBI_PATH,
        tracciati_path=TRACCIATI_PATH,
    )
    n = len(ds_total)
    print(f"[OK]   Pool totale: {n} campioni "
          f"(Aariz={ds_total.len_aariz} | "
          f"ISBI={ds_total.len_isbi} | "
          f"Tracciati={ds_total.len_tracciati})")

    # Verifica campioni distribuiti nel pool
    n_err = 0
    step  = max(1, n // 50)
    for i in range(0, n, step):
        img, pts, mask, bbox, px = ds_total[i]
        validi = pts[mask == 1.0]
        if len(validi) == 0:
            continue
        if validi.max().item() > 1.001 or validi.min().item() < -0.001:
            n_err += 1
            if n_err <= 3:
                print(f"  [POOL ERR] idx={i}: min={validi.min():.4f} "
                      f"max={validi.max():.4f}")

    if n_err == 0:
        print(f"[OK]   Pool: coordinate in [0,1] verificate su {n//step} campioni")
    else:
        print(f"[FAIL] Pool: {n_err} campioni con coordinate fuori [0,1]")
        tutti_ok = False

except Exception as e:
    print(f"[FAIL] Pool aggregato: {e}")
    tutti_ok = False

# ── RISULTATO FINALE ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if tutti_ok:
    print("✓ Tutti i check OK — puoi lanciare il training.")
else:
    print("✗ Ci sono errori — correggi prima di lanciare il training.")
    print("\nPossibili cause:")
    print("  - coord_scale=0.01 errato per alcune cartelle Tracciati")
    print("  - Normalizzazione mancante in uno dei dataset")
    print("  - File Excel con coordinate in unità diverse")
    sys.exit(1)