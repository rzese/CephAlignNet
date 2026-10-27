# CephAlignNet — Documentazione tecnica

CephAlignNet individua automaticamente **26 landmark cefalometrici** su teleradiografie
laterali del cranio. È una pipeline PyTorch in due stadi:

1. **Stage 1**: una rete HRNet-W48 predice per ogni landmark una *heatmap* (mappa di
   probabilità) più due mappe di *offset*. Dalla heatmap si ricavano le coordinate.
2. **Stage 2** (opzionale): una piccola CNN con attenzione guarda un ritaglio delle
   feature dello Stage 1 attorno a ogni punto predetto e stima una piccola correzione
   `(dx, dy)`. Viene applicata solo ai landmark più difficili.

Questo documento spiega cosa fa ogni file, come scorrono i dati, come si addestra,
valuta e usa il modello, e quali limiti e problemi noti ha il codice attuale.
Il `README.md` (in inglese) resta la panoramica breve.

---

## Indice

1. [Struttura della repository](#1-struttura-della-repository)
2. [I 26 landmark](#2-i-26-landmark)
3. [Convenzioni sulle coordinate](#3-convenzioni-sulle-coordinate)
4. [Dataset](#4-dataset)
5. [Stage 1 — HRNetLandmark](#5-stage-1--hrnetlandmark)
6. [Funzioni di supporto (`heatmap_utils.py`)](#6-funzioni-di-supporto-heatmap_utilspy)
7. [Training dello Stage 1 (`train_heatmap.py`)](#7-training-dello-stage-1-train_heatmappy)
8. [Statistiche per landmark (`analiz_land.py` → `landmark_stats.py`)](#8-statistiche-per-landmark)
9. [Stage 2 — RefinementNet](#9-stage-2--refinementnet)
10. [Training dello Stage 2 (`train_refine.py`)](#10-training-dello-stage-2-train_refinepy)
11. [Valutazione (`evaluate_refine.py`)](#11-valutazione-evaluate_refinepy)
12. [Inferenza su un'immagine (`predict_heatmap.py`)](#12-inferenza-su-unimmagine-predict_heatmappy)
13. [Cartella `Test/`](#13-cartella-test)
14. [Come si esegue](#14-come-si-esegue)
15. [Risultati dell'ultimo training](#15-risultati-dellultimo-training)
16. [Problemi noti e correzioni](#16-problemi-noti-e-correzioni)
17. [Sigma addestrabile e incertezza](#17-sigma-addestrabile-e-incertezza)

---

## 1. Struttura della repository

```
CephAlignNet/
├── train_heatmap.py        # training Stage 1
├── train_refine.py         # training Stage 2 (Stage 1 congelato)
├── evaluate_refine.py      # MRE / SDR di Stage 1 e Stage 1+2
├── predict_heatmap.py      # inferenza su una singola immagine
├── src/
│   ├── model_hrnet.py      # Stage 1: HRNet-W48 + testa di fusione
│   ├── model_refine.py     # Stage 2: RefinementNet (CBAM)
│   ├── heatmap_utils.py    # heatmap, loss, decodifica, augmentation
│   ├── incertezza.py       # sigma addestrabile, momenti della heatmap, calibrazione
│   ├── inferenza.py        # caricamento modelli, TTA, Stage 2, collate: condivisi
│   ├── dataset.py          # AarizDataset
│   ├── dataset_isbi.py     # ISBIDataset (Cepha400 / ISBI 2015)
│   ├── dataset_tracciati.py# TracciatiDataset (tracciati clinici in Excel)
│   ├── datasets.py         # Datasets: unisce i tre dataset
│   ├── analiz_land.py      # calcola l'MRE per landmark e genera landmark_stats.py
│   └── landmark_stats.py   # FILE GENERATO: maschera Stage 2, pesi, sigma
└── Test/
    ├── test_unita.py               # test unitari su dati sintetici (CPU)
    ├── verifica_mappatura_isbi.py  # controllo visivo della mappatura ISBI
    └── ...                         # script di verifica visiva
```

`.gitignore` esclude `data/` e tutti i `*.pth`: dataset e pesi non sono versionati.

### Flusso complessivo

```
Aariz (JSON) ─┐
ISBI (TXT)  ──┼─► Datasets ─► collate (resize 512, norm [0,1], augment, heatmap GT)
Tracciati   ──┘                         │
(Excel, opz.)                           ▼
                           train_heatmap.py ──► modello_heatmap.pth
                                         │
                                         └─► analiz_land ──► src/landmark_stats.py
                                                             (REFINE_LANDMARK_MASK,
                                                              PESI_LANDMARK,
                                                              SIGMA_PER_LANDMARK)
                                         ▼
                           train_refine.py ──► modello_refine.pth
                                         ▼
           evaluate_refine.py (MRE/SDR)    predict_heatmap.py (immagine → coordinate)
```

---

## 2. I 26 landmark

L'ordine canonico è definito in `SYMBOL_ORDER` (`src/dataset.py`) e ripetuto come
`NOMI_26`, `NOMI` e `NOMI_LANDMARK` in altri file:

| Idx | Nome | Idx | Nome | Idx | Nome |
|---|---|---|---|---|---|
| 0 | A (subspinale) | 9 | R (rhinion) | 18 | LMT (molare inferiore) |
| 1 | ANS | 10 | S (sella) | 19 | UIT (punta incisivo sup.) |
| 2 | B (supramentale) | 11 | Ar (articulare) | 20 | UIA (apice incisivo sup.) |
| 3 | Me (menton) | 12 | Co (condylion) | 21 | LIA (apice incisivo inf.) |
| 4 | N (nasion) | 13 | Gn (gnathion) | 22 | Ls (labbro sup.) |
| 5 | Or (orbitale) | 14 | Go (gonion) | 23 | Li (labbro inf.) |
| 6 | Pog (pogonion) | 15 | Po (porion) | 24 | Pog\` (pogonion molle) |
| 7 | PNS | 16 | LPM (premolare inf.) | 25 | Sn (subnasale) |
| 8 | Pn (pronasale) | 17 | LIT (punta incisivo inf.) | | |

- **Aariz** ha tutti i 26 landmark.
- **ISBI** ne fornisce 19 (S, N, Or, Po, A, B, Pog, Me, Gn, Go, LIT, UIT, Ls, Li, Sn,
  Pog\`, PNS, ANS, Ar). Gli altri (Pn, R, Co, LPM, LMT, UIA, LIA) vengono mascherati.

> Nota: questo set **non** coincide con i 35 landmark usati negli script di overlay
> della cartella superiore (`overlay_punti.py`: `SOr`, `Na`, `U1Inc`, `L6M`, `G`, …).
> La traduzione dei nomi è in `NAME_MAP` di `dataset_tracciati.py` (§4.3).

---

## 3. Convenzioni sulle coordinate

| Contesto | Unità | Origine / range |
|---|---|---|
| Output dei dataset | **pixel dell'immagine originale** | `(0,0)` in alto a sinistra, y verso il basso |
| Punto non annotato | `[0.0, 0.0]` | diventa `mask = 0` in `Datasets` |
| Dentro i `collate_fn` e nei modelli | **normalizzate [0, 1]** | x / larghezza, y / altezza |
| Input della rete | 512 × 512, 1 canale, valori [0, 1] | resize **senza** mantenere le proporzioni |
| Heatmap | 256 × 256 (`HEATMAP_TRAIN_SIZE`) | |
| Feature map per lo Stage 2 | 240 canali × 256 × 256 | 5 rami × 48 canali |
| Offset Stage 2 | normalizzati, limitati a ±`MAX_OFFSET` | |

**Conversione in mm** (solo in valutazione):
`errore_mm = √((Δx·W)² + (Δy·H)²) × pixel_size`, dove W e H sono le dimensioni
originali dell'immagine e `pixel_size` è in mm/px (`errori_mm` in `src/inferenza.py`).

---

## 4. Dataset

Tutti i dataset restituiscono l'immagine come tensore `float [1, H, W]` in [0, 1],
caricata in scala di grigi con OpenCV.

### 4.1 `AarizDataset` (`src/dataset.py`)

Struttura attesa:

```
data/Aariz/
├── cephalogram_machine_mappings.csv           # colonne cephalogram_id, pixel_size
└── {train,valid,test}/
    ├── Cephalograms/<id>.png|jpg|jpeg|bmp
    └── Annotations/Cephalometric Landmarks/
        ├── Senior Orthodontists/<id>.json
        └── Junior Orthodontists/<id>.json     # opzionale
```

- `split='all'` (usato da `Datasets`) unisce train, valid e test.
- Un campione viene tenuto solo se esiste il JSON *Senior*.
- `_parse_json` legge i punti **per simbolo** (`item['symbol']`), quindi non dipende
  dall'ordine nel file. I simboli mancanti diventano `[0, 0]`.
- Se c'è anche il JSON *Junior*, le coordinate sono `ceil((senior + junior) / 2)`.
- `pixel_size` viene da `cephalogram_machine_mappings.csv`; se manca, vale 0.1 mm/px.
- Ritorna `(image, landmarks[26,2], pixel_size)`.

### 4.2 `ISBIDataset` (`src/dataset_isbi.py`)

Struttura attesa:

```
data/Cepha400/
├── RawImage/{TrainingData (001-150), Test1Data (151-300), Test2Data (301-400)}/<nnn>.bmp
└── AnnotationsByMD/{400_junior,400_senior}/<nnn>.txt
```

- `parse_md_txt` legge le prime 19 righe `x,y` di ogni TXT. Le righe 20–27 (punteggi
  di qualità) vengono ignorate.
- Le coordinate sono la media tra junior e senior.
- P1…P19 vengono rimappati sugli indici canonici con una delle due mappature in
  `MAPPATURE_ISBI`, scelta da `MAPPATURA_ISBI_DEFAULT`:
  - `"standard"` (default): l'ordine pubblicato per la challenge ISBI 2015.
    **Verificata visivamente** sui casi 001, 151 e 301.
  - `"legacy"`: quella usata prima. Solo P1–P4 erano corretti; resta solo per
    riprodurre vecchi esperimenti.

  **Le due mappature non coincidono dal P5 in poi** (vedi §16, punto 5).
- Se l'immagine non si legge, viene usato un placeholder nero 1935×2400.
- Ritorna `(image, landmarks[26,2])` senza `pixel_size`: `Datasets` assegna 0.1 mm/px.

### 4.3 `TracciatiDataset` (`src/dataset_tracciati.py`)

È il dataset interno dei tracciati clinici: un'immagine più un file Excel per caso.

```
<base_path>/<sottocartella>/<case_id>/
    <case_id>.jpg        # immagine intera (o un'altra immagine come fallback)
    <case_id>-T.jpg      # miniatura/tracciato (usata solo con use_thumbnail=True)
    *.xlsx | *.xltx | *.xls
```

- **`_identifica_file`** cerca nella cartella del caso l'immagine e il primo file
  Excel. L'immagine preferita è quella con nome uguale al `case_id`. I file `*-T.*`
  sono considerati miniature.
- **`_parse_excel`**:
  1. Scorre tutte le celle di ogni foglio.
  2. Se il testo di una cella è in `NAME_MAP` (provando anche le versioni
     *Capitalize* e *UPPER*), prende come `(x, y)` i primi due numeri successivi
     sulla stessa riga.
  3. Moltiplica per `coord_scale` e, se richiesto, ribalta la y.
  4. Usa il primo foglio in cui trova almeno un landmark.
- **`NAME_MAP`** contiene circa 300 varianti di nome (italiano/inglese, sigle, maiuscole
  diverse, nomenclatura ImageJ come `U1Inc`, `L6M`, `U1Ap`) mappate sui 26 indici.
- **`__getitem__`**: se Gn manca ma Me e Pog sono presenti, calcola Gn come loro
  punto medio.
- **Diagnostica**: con `DEBUG = True`, oppure eseguendo il modulo direttamente
  (`python -m src.dataset_tracciati <cartella>`), stampa il contenuto grezzo dell'Excel
  e l'elenco dei landmark trovati e mancanti, così si possono aggiungere i nomi
  mancanti a `NAME_MAP`.

- **Scala per caso** (px/mm). Viene cercata in quest'ordine:
  1. nella tabella `tabella_scale`, un CSV o Excel con colonne `case_id` e
     `scale_px_per_mm` (come `coordinate_radiografie.xlsx`);
  2. in un `.docx` di ImageJ nella cartella del caso ("Distance in pixels … Known
     distance …").

  Se la scala è nota, i valori Excel vengono convertiti in pixel
  (mm × 1000 per i file ImageJ, riconosciuto in automatico con `unita_excel="auto"`)
  e il caso riceve `pixel_size = 1/scala`. Altrimenti si usa il fattore globale
  `coord_scale` e viene stampato un avviso.
- Restituisce `(image, landmarks, pixel_size)` e segnala i punti che finiscono fuori
  dall'immagine.

Come viene istanziato da `Datasets`:
- `subfolder_names` = `tracciati_subfolders` (default `["tracciato 50", "tracciato 50_2"]`)
- `coord_scale=0.01` (solo come fallback)
- `tabella_scale` = `tracciati_tabella_scale`
- `flip_y=False`, `pixel_size=0.1`, `use_thumbnail=False`

### 4.4 `Datasets` (`src/datasets.py`)

Unisce i tre dataset in un unico `Dataset`, con indici consecutivi nell'ordine
Aariz, poi ISBI, poi Tracciati. Ogni campione esce come:

```
(image [1,H,W], points [26,2] px, mask [26], bbox [4], pixel_size)
```

- `mask[i] = 0` se il punto vale esattamente `(0, 0)`.
- `bbox` è il rettangolo che contiene i punti validi, con 20 px di margine e limitato
  all'immagine. Non viene usato dagli script di training.
- I Tracciati si caricano solo se `tracciati_path` esiste.
- **`crea_split(dataset, val_split, seed, modo)`**: è lo split train/validation usato da
  *tutti* gli script. Due modalità, scelte con la variabile d'ambiente `CEPH_SPLIT`:
  - **`ufficiale`** (default): la divisione con cui arrivano i dataset. Aariz `train` →
    training, `valid` → validation, `test` → test; ISBI `TrainingData` → training, tranne
    un'immagine ogni 5 (005, 010, …, 150: 30 immagini) → validation; ISBI `Test1Data` e
    `Test2Data` → test. I Tracciati vanno tutti nel training. Risultato: 820 training,
    180 validation, 400 test (150 Aariz + 150 ISBI Test1 + 100 ISBI Test2).
  - **`casuale`**: il vecchio split 80/20 con seed 42 su Aariz + ISBI, solo per
    rivalutare i modelli addestrati con quello (per esempio `esperimenti_leo`).

  Il validation serve per early stopping, taratura della fusione, soglie e calibrazione
  dell'incertezza; i test set solo per i risultati finali.
- **`crea_insieme(dataset, nome)`**: le immagini su cui valutare: `validation` oppure
  `test`, `test_aariz`, `test_isbi1`, `test_isbi2` (solo con lo split ufficiale). Usato da
  `evaluate_refine.py`, `analisi_stage2.py` e `coda_errori.py` (opzione `--insieme`).
- **Impronta del validation set**: `crea_split` stampa un codice breve delle immagini di
  validation; i checkpoint lo salvano e, se al caricamento non coincide, compare un avviso
  ("results are NOT valid").
- `ceph_path`/`ceph_base` sono ignorati: CephBiometrics è stato rimosso perché
  duplicava ISBI.
- Il percorso ISBI di default qui è `data/ISBI2015`, ma tutti gli script passano
  `data/Cepha400`.

---

## 5. Stage 1 — `HRNetLandmark` (`src/model_hrnet.py`)

```
input [B,1,512,512]
   │  timm "hrnet_w48", features_only=True, in_chans=1
   ▼
k feature map multi-scala (con questo checkpoint 5 rami: 64,128,256,512,1024 canali)
   │  per ramo: Conv1x1 → 48 canali + BN + ReLU, poi upsample bilineare alla risoluzione
   │  del ramo più fine
   ▼
concat → x_out [B, k·48, 256, 256]   (= 240 canali)  ──► restituito per lo Stage 2
   │  Conv3x3 128 → Conv3x3 64 → Conv1x1 (26·3)
   ▼
pred [B, 78, 256, 256]
```

Per ogni landmark `i`, i canali dell'output sono:

- `3i`: heatmap
- `3i+1`: offset x
- `3i+2`: offset y

Gli offset sono normalizzati con raggio `R = 40` px della heatmap.

- `forward` ritorna `(pred, x_out)`.
- `hrnet_channels=None` fa un forward con un tensore fittizio per scoprire le
  larghezze dei canali.
- Gli script di caricamento (`carica_stage1`) provano varie liste di canali finché
  `load_state_dict(strict=True)` funziona. Per questo il log mostra più righe
  `Backbone: ... Detected channels`.

---

## 6. Funzioni di supporto (`heatmap_utils.py`)

### Generazione del ground truth
- **`crea_heatmap_gaussiana(h, w, cx, cy, sigma)`**: gaussiana 2D non normalizzata
  (picco = 1).
- **`coordinate_a_heatmap(pts, mask, …)`**: una heatmap 256×256 per ogni landmark
  valido. La sigma di ogni landmark viene da `SIGMA_PER_LANDMARK` (default 4 px);
  i landmark più difficili hanno heatmap più larghe.

### Decodifica (heatmap → coordinate)
- **`heatmap_a_coordinate(h, temperatura=100)`**: *soft-argmax*, cioè
  `softmax(h·T)` pesato su una griglia [0,1]. È differenziabile ed è il default
  ovunque.
- **`regression_voting(output, N, R=40)`**: prende i `π·R²` pixel con heatmap più
  alta, ciascuno "vota" con `posizione + offset·R`, e il risultato è la media dei
  voti. Si attiva solo in `predict_heatmap.py` con `USE_VOTING = True`.
- `heatmap_a_coordinate_legacy`: vecchia variante, non usata.

### Loss
| Funzione | Cosa misura |
|---|---|
| `awing_loss` | Adaptive Wing loss (Wang et al. 2019) sulle heatmap, pesata per landmark con `PESI_LANDMARK` e mascherata |
| `combined_loss` | `2/3 · AWing(heatmap) + 1/3 · L1(offset)`; la L1 conta solo i pixel entro R=40 dal punto vero |
| `wing_loss_coord` | Wing loss sulle coordinate decodificate con soft-argmax (`w=10, eps=2` su valori in [0,1]: in pratica si comporta come una loss logaritmica) |
| `geometric_consistency_loss` | 8 vincoli anatomici (vedi sotto) |

Vincoli geometrici (penalità quadratica solo quando sono violati):

- Gn vicino al punto medio di Me e Pog
- Me sotto B, e B sotto A
- ANS davanti a PNS
- Co sopra Go
- S dietro N
- UIT (punta dell'incisivo superiore) sotto UIA (apice), e LIA sotto LIT

"Davanti" e "dietro" si riferiscono al verso del volto, ricavato dal ground truth di
ogni campione (segno di N_x − S_x): i vincoli valgono quindi anche sulle immagini
ribaltate.

### Augmentation
- **`crop_campo_augmenta(img, pts, mask, pixel_size, margine_mm=5)`**: ritaglio del
  campo di vista sull'immagine **originale**, prima del resize a 512, per rendere
  lo Stage 1 robusto alle radiografie "zoommate" (campo stretto, come parte dei
  tracciati UNIFE). Ogni lato viene tagliato di una quantità casuale fra 0 e il
  massimo che lascia **tutti** i landmark annotati ad almeno 5 mm dal nuovo bordo;
  i lati sono indipendenti, quindi cambia anche il rapporto d'aspetto. In training
  si applica con p = 0.5 (`crop_campo_prob`, `crop_campo_margine_mm` nel CONFIG di
  `train_heatmap.py`; p = 0 lo disattiva). Su 60 immagini di training la
  larghezza mediana del campo passa da 194 a 158 mm.
- **`augmenta(img, pts, mask)`**, applicata in training a ogni campione:
  - flip orizzontale (p = 0.5)
  - rotazione ±20° (sempre)
  - traslazione ±8% (p = 0.6)
  - scala 0.85–1.15 con crop/pad (p = 0.6)
  - luminosità e contrasto ×[0.65, 1.35]
  - gamma 0.7–1.4 (p = 0.5)
  - rumore gaussiano σ ∈ [0.01, 0.04]
  - blur 3×3 (p = 0.3)

  Un punto che esce dall'immagine viene **mascherato**, non schiacciato sul bordo.
- **`zoom_crop_augmenta(…, zoom_factor=2.2)`**: ritaglia una finestra larga
  1/2.2 dell'immagine, centrata sulla media dei landmark difficili
  (Or, S, Po, LPM, UIT, UIA, LIA), e la riporta a 512×512. In training si applica
  con p = 0.4 dopo `augmenta`.

---

## 7. Training dello Stage 1 (`train_heatmap.py`)

**`CONFIG`** principale:

| Parametro | Valore |
|---|---|
| `img_size` | 512 |
| `heatmap_size` | 256 |
| `sigma` | 4.0 |
| `batch_size` | 2 (nel log dell'ultimo run: 16) |
| `num_epochs` | 600 |
| `learning_rate` | 3e-5 |
| `patience` | 60 |
| `geo_loss_weight` | 0.05 |
| `val_split` | 0.2 (seed 42, solo con `CEPH_SPLIT=casuale`) |
| `save_path` | `modello_heatmap.pth` |

Passi:

1. **`prepara_dati`**: crea `Datasets` (con i Tracciati se la cartella esiste) e usa
   `crea_split` (split ufficiale per default).
2. **`collate_con_heatmap`**:
   - resize a 512×512
   - normalizzazione delle coordinate in [0,1]
   - padding a 26 landmark
   - in training: `crop_campo_augmenta` con p = 0.5 (sull'immagine originale, prima
     del resize), poi `augmenta`, più `zoom_crop_augmenta` con p = 0.4
   - creazione delle heatmap target
3. **Loss totale**:
   `0.7·combined_loss + (0.3 − geo_w)·wing_loss_coord + geo_w·geometric_consistency_loss`
4. **Ottimizzazione**:
   - AdamW (weight decay 1e-4)
   - gradient clipping a 3.0
   - `ReduceLROnPlateau` (patience 20, fattore 0.5, min 5e-7)
   - early stopping dopo `patience` epoche senza miglioramento
5. Salva il checkpoint migliore:
   `{epoca, model_state_dict, optimizer_state_dict, val_loss, config, model_name, backbone}`.
6. A fine training chiama `genera_landmark_stats()` per rigenerare
   `src/landmark_stats.py` (§8).

---

## 8. Statistiche per landmark

`src/analiz_land.py` valuta lo Stage 1 sul validation set (senza TTA e senza Stage 2),
calcola l'MRE di ogni landmark e scrive `src/landmark_stats.py` con tre dizionari:

| Dizionario | Regola | Usato da |
|---|---|---|
| `REFINE_LANDMARK_MASK` | landmark con MRE > 1.5 mm | `evaluate_refine.py`, `predict_heatmap.py` (decide su quali punti si applica lo Stage 2) |
| `PESI_LANDMARK` | `1 + 1.0·(MRE − 1)`, solo se MRE > 1 mm, massimo 4 | `awing_loss` |
| `SIGMA_PER_LANDMARK` | `4 + 0.8·(MRE − 1)` px, massimo 7 | `coordinate_a_heatmap` |

Se `landmark_stats.py` non esiste, `heatmap_utils.py` e `model_refine.py` usano valori
di fallback scritti nel codice e stampano un `[WARN]`.

**Il ciclo è iterativo**: il primo training usa i valori di fallback; le statistiche
prodotte alla fine servono per un training successivo. La maschera Stage 2 invece
cambia subito, perché viene letta a ogni import.

Esecuzione manuale dalla root della repo:

```bash
python -m src.analiz_land
```

Il messaggio di warning parla di `analyze_landmark_difficulty.py`, ma quel file non
esiste: lo script vero è `src/analiz_land.py`.

---

## 9. Stage 2 — `ImageRefinementNet` (`src/model_refine.py`)

Lo Stage 2 corregge la posizione di un punto guardando un **ritaglio della radiografia
originale**, alla risoluzione piena, attorno alla predizione dello Stage 1.

> **Perché è cambiato.** Il vecchio Stage 2 (`RefinementNet`, ancora caricabile per
> valutare i checkpoint vecchi) guardava un ritaglio delle *feature dello Stage 1*.
> Sulle immagini di training quelle feature "sanno" già dov'è il punto vero, perché lo
> Stage 1 le ha viste centinaia di volte; sulle immagini nuove contengono lo stesso
> errore dello Stage 1. Il risultato era una correzione dalla direzione casuale: secondo
> `analisi_stage2.py` migliorava il 48% dei punti e ne peggiorava il 52%.

```
radiografia originale (≈0.1 mm/px)
     │  ritaglio 38.4 × 38.4 mm centrato sul punto dello Stage 1
     ▼  (roi_align → 192 × 192 px = 0.2 mm/px, normalizzato media 0 / dev. std 1)
encoder CNN (192 → 48 celle, blocchi residui con dilatazione 1-2-4-8)
     ▼
26 heatmap 32×32 (una per landmark) → si sceglie quella del landmark
     ▼  softmax + baricentro (soft-argmax)
posizione del landmark nel ritaglio → offset (dx, dy) in mm dal centro
```

- **Dimensioni in mm, non in pixel**: la finestra è sempre di 38.4 mm (192 px, 0.2 mm/px), qualunque sia il
  pixel size dell'immagine, quindi i ritagli sono confrontabili tra dataset diversi.
- **Heatmap + baricentro invece di una regressione diretta**: la rete indica *dove* è il
  punto nel ritaglio. Le convoluzioni conservano l'informazione spaziale, che il vecchio
  regressore con pooling globale perdeva.
- **Correzione massima** `max_offset_mm` (default 11.5 mm). In applicazione le correzioni
  oltre 1.2 × `max_offset_mm` per asse vengono scartate (guardia).

## 10. Training dello Stage 2 (`train_refine.py`)

**Non serve lo Stage 1.** Per ogni landmark annotato:

1. si parte dal **punto vero** e lo si sposta di un **errore sintetico**: distanza
   **uniforme** tra `rumore_min_mm` = 0.5 e `rumore_max_mm` = 11.5 mm, direzione
   uniforme. Non si fa nessuna ipotesi sulla distribuzione degli errori reali dello
   Stage 1: con la fusione lo Stage 2 pesa soprattutto dove lo Stage 1 è incerto, quindi
   deve funzionare allo stesso modo a ogni distanza;
2. si ritaglia la radiografia attorno al punto spostato;
3. la rete deve indicare dov'è il punto vero nel ritaglio.

Così lo Stage 2 non può "copiare" lo Stage 1: deve riconoscere l'anatomia.

- **Augmentation**: rotazione ±10° del ritaglio (con il bersaglio ruotato di
  conseguenza), contrasto e luminosità, rumore, blur occasionale.
- **Loss**: Huber sull'offset in mm + `peso_nll` × NLL della posizione vera sulla
  heatmap del ritaglio (come la modalità `probabilistic` dello Stage 1).
- **Validazione**: stessi errori sintetici, ma con seed fisso, quindi confrontabili tra
  epoche. Il log riporta l'errore medio e mediano **prima → dopo** lo Stage 2 e la
  percentuale di punti migliorati: è il modo più diretto per vedere se sta imparando.
- AdamW (LR 1e-3), `ReduceLROnPlateau`, early stopping (patience 30). Il checkpoint
  salva `tipo_input = "immagine"`, `finestra_mm`, `crop_px` e `max_offset_mm`, che
  `carica_stage2` usa per ricostruire la rete.

**A quali punti si applica** (`selezione_stage2` in `evaluate_refine.py` e
`predict_heatmap.py`):

| Valore | Punti corretti |
|---|---|
| `"fusione"` (default) | tutti, ma il punto finale è una **media pesata** di Stage 1 e Stage 2: `S1 + w·(S2 − S1)` con `w = σ_tta² / (σ_tta² + (c·σ_S2)²)`, dove σ_S2 è la larghezza della heatmap dello Stage 2 nel ritaglio (`fusione_stage2`: c = 0.32, p = 2). Richiede la TTA. Risultato in `strategie_stage2.py` (validazione incrociata pari/dispari): accuratezza < 2 mm 86.6% → 88.4%, MRE 1.085 → 1.029 mm |
| `"incertezza"` | quelli con `sigma_tta` > `soglia_sigma_tta_mm` (default 0.25 mm, circa il 50% più incerto per il modello `probabilistic`: soglia scelta sulle immagini pari e verificata sulle dispari, MRE 1.099 → 1.060 mm; richiede la TTA) |
| `"tutti"` | tutti i landmark |
| `"maschera"` | quelli in `REFINE_LANDMARK_MASK` (`src/landmark_stats.py`) |

I parametri (modalità, `c`, `p`, `floor`, soglia) stanno in **`src/parametri_stage2.py`**,
letto da `evaluate_refine.py` e `predict_heatmap.py`. Come per `src/landmark_stats.py`,
il primo run serve da misura empirica e i run successivi usano quei valori **fissi**:
il file non viene rigenerato automaticamente. I valori attuali vengono dal primo run
`probabilistic` (c = 0.32, p = 2, floor = 0, soglia 0.25 mm).

Secondo run, con lo Stage 1 che parte dalle statistiche per landmark del primo
(`STATS_DA/<modalità>/landmark_stats.py`) e risultati in una cartella separata:

```bash
sbatch --export=ALL,MODES=probabilistic,STATS_DA=esperimenti,OUT=esperimenti_run2 run_tutto.slurm
```

Per rimisurare i parametri su un run (solo quando si vuole sostituire la misura):

1. `analisi_stage2.py --dir <esperimento>` (GPU) salva per ogni punto errori,
   spostamenti proposti dallo Stage 2 (anche con la sua TTA) e incertezze;
2. `strategie_stage2.py <csv>` (solo numpy) confronta solo Stage 1, Stage 2 ovunque,
   soglia, fusione e classificatore, scegliendo i parametri per massimizzare
   l'accuratezza entro 2/2.5/3/4 mm su metà delle immagini e misurandola sull'altra
   metà (e viceversa).
3. `strategie_stage2.py <csv> --salva src/parametri_stage2.py` riscrive il file con i
   parametri stimati su tutte le immagini.

---

## 11. Valutazione (`evaluate_refine.py`)

- Usa Aariz + ISBI (**i Tracciati sono esclusi di proposito**) e prende il 20% di
  validazione con seed 42.
- **TTA** (`predici_con_tta`), media di 8 predizioni:
  - 5 angoli (−5, −2.5, 0, +2.5, +5°), ognuno de-ruotato
  - 1 flip orizzontale, con x ribaltata
  - 2 varianti fotometriche (luminosità/contrasto ×[0.88, 1.12])

  `x_out` viene preso dal passaggio non aumentato (angolo 0). Il codice è in
  `src/inferenza.py`, condiviso con `predict_heatmap.py`.
- **`applica_stage2`**: corregge solo i landmark in `REFINE_LANDMARK_MASK`, con un
  solo forward per batch. Scarta la correzione se `|dx|` o `|dy|` superano la guardia
  `1.2·max_offset`, dove `max_offset` è quello del checkpoint.
- **`calcola_mre_sdr`**:
  - errore in mm calcolato con `errori_mm` (x scalata per la larghezza, y per l'altezza)
  - **MRE**: media degli errori, globale e per landmark
  - **SDR@s**: % di errori sotto 2, 2.5, 3 e 4 mm
- **`stampa_risultati`**: report globale, per categoria (scheletrici, dentali,
  tessuti molli) e tabella per landmark. Alla fine stampa il confronto
  Stage 1 → Stage 1+2.
- **Incertezza** (`analizza_incertezza=True`): analisi delle tre fonti di incertezza
  e scrittura di `calibrazione_incertezza.json` (§17).

---

## 12. Inferenza su un'immagine (`predict_heatmap.py`)

```bash
python predict_heatmap.py percorso/radiografia.png 0.1
```

Il secondo argomento, opzionale, è il pixel size in mm/px e serve per esprimere
l'incertezza in mm.

1. Carica lo Stage 1 e, se `usa_stage2 = True` e il file esiste, lo Stage 2.
2. Legge l'immagine in scala di grigi e la ridimensiona a 512×512.
3. Esegue lo Stage 1 con TTA (`usa_tta`) e decodifica con soft-argmax, oppure con
   regression voting se `USE_VOTING = True`.
4. Applica lo Stage 2 ai landmark della maschera, con la guardia 1.2·MAX_OFFSET.
5. Riporta le coordinate alla dimensione originale (`x·W`, `y·H`).
6. Stampa per ogni landmark le coordinate in pixel, σ_heatmap e σ_tta in mm e, se
   esiste `calibrazione_incertezza.json`, il raggio calibrato r90.
7. Salva `<immagine>_pred.png` (`salva_overlay`) con i punti e i cerchi r90, e apre
   la finestra OpenCV solo se `mostra_finestra = True` (sui server senza display va
   messo a `False`).

Senza argomento usa un'immagine Aariz di esempio scritta nel codice.

---

## 13. Cartella `Test/`

Tutti gli script vanno eseguiti dalla root della repo con `PYTHONPATH=.`.

| File | Cosa fa |
|---|---|
| `test_unita.py` | **24 test automatici** su dati sintetici, senza dataset né GPU: rotazione TTA, coerenza di `augmenta`, griglia pixel, regression voting, sigma addestrabile, loss NLL, momenti della heatmap, conversione in mm, vincoli geometrici, Stage 2 e guardia, split, scala dei Tracciati, calibrazione conforme. Si lancia con `PYTHONPATH=. python Test/test_unita.py`, oppure con `pytest`. |
| `verifica_mappatura_isbi.py` | Salva alcune immagini ISBI con i 19 punti etichettati secondo le due mappature, per scegliere quella giusta |
| `test_caricamento tracciati.py` | Legge ogni caso dei Tracciati e riporta quanti landmark riconosce |
| `test.py`, `test_isbi.py`, `test_datasets.py`, `test_tracciati.py` | Visualizzazione di un campione; aggiornati all'interfaccia attuale dei dataset |
| `test_ceph.py` | Non più eseguibile: `src/dataset_ceph.py` è stato rimosso |

---

## 14. Come si esegue

### Requisiti
- Python 3, `torch`, `torchvision`, `timm`
- `opencv-python`, `numpy`, `pandas`, `openpyxl` (solo per i Tracciati)
- GPU CUDA consigliata per il training

### Ordine dei comandi (dalla root della repo)

```bash
python train_heatmap.py
```
```bash
python train_refine.py
```
```bash
python evaluate_refine.py
```
```bash
python predict_heatmap.py data/Aariz/test/Cephalograms/<id>.png
```

Tutti i percorsi e gli iperparametri si cambiano nel dizionario `CONFIG` in cima a
ogni script: non ci sono argomenti da riga di comando, tranne il percorso
dell'immagine in `predict_heatmap.py`. I `.pth` vengono letti e scritti nella
directory corrente.

---

## 15. Risultati dell'ultimo training

Dati presi da `../model_no_unife/results.txt`:

- 1400 immagini (Aariz 1000 + ISBI 400), **Tracciati = 0**
- Stage 1: miglior checkpoint all'epoca 385 (early stop all'epoca 445)
- Stage 2: early stop all'epoca 320

| Metrica (con TTA) | Stage 1 | Stage 1 + 2 |
|---|---|---|
| MRE globale | **1.47 mm** | 1.59 mm |
| SDR 2 mm | **77.9%** | 74.0% |
| SDR 2.5 mm | 85.3% | 82.4% |
| SDR 3 mm | 90.0% | 87.5% |
| SDR 4 mm | 95.1% | 93.8% |

**Lo Stage 2 peggiora i risultati.** Le possibili cause sono in §16 (punti 1–3).

Il `src/landmark_stats.py` presente nella repo **non** viene da questo run: elenca 10
landmark per lo Stage 2, mentre il log ne riporta 12
(`[1, 5, 7, 9, 11, 12, 14, 15, 16, 18, 20, 21]`). Dopo aver cambiato i pesi conviene
rigenerarlo (§8).

---

## 16. Problemi noti e correzioni

| # | Problema | Stato |
|---|---|---|
| 1 | **La TTA de-ruotava nel verso sbagliato.** Le predizioni sulle immagini ruotate venivano ruotate *ancora* di ±angolo invece di essere riportate indietro. | **Corretto** (`ruota_coordinate`, `src/inferenza.py`). Verificato con un test numerico: con 5° il codice vecchio sbagliava di circa 0.06 in coordinate normalizzate. |
| 2 | **`MAX_OFFSET` non corrispondeva al checkpoint.** Lo Stage 2 salvato con 0.016 veniva usato con 0.008, quindi le correzioni erano dimezzate. | **Corretto**: `carica_stage2` legge `max_offset` dal checkpoint. |
| 3 | **Lo Stage 2 è addestrato senza TTA ma usato con la TTA.** | Non modificato: va rivalutato dopo la correzione 1, che cambia la distribuzione degli errori in ingresso. |
| 4 | **Il flip contraddiceva la loss geometrica.** I vincoli orizzontali (S/N, ANS/PNS) erano sbagliati sulle immagini ribaltate. | **Corretto**: il verso del volto si ricava dal ground truth di ogni campione. |
| 5 | **Mappatura ISBI sbagliata**: 15 punti su 19 avevano l'etichetta sbagliata su tutte le 400 immagini (ad esempio il punto A era addestrato come Ar e il gonion come Pog). | **Corretto**: `MAPPATURA_ISBI_DEFAULT = "standard"`, verificata visivamente con `Test/verifica_mappatura_isbi.py`. Richiede un nuovo training. |
| 6 | **La scala dei Tracciati era fissa** (10 px/mm per tutte le immagini). | **Corretto**: scala per caso da tabella o da `.docx`; il fallback globale stampa un avviso. |
| 7 | **La validazione cambiava se si attivavano i Tracciati**, con il rischio di valutare su immagini di training. | **Corretto** con `crea_split`. |
| 8 | **La conversione in mm usava 2400 px per entrambi gli assi.** | **Corretto**: `errori_mm` usa larghezza, altezza e pixel size di ogni immagine. |
| 9 | **Categorie sbagliate nel report** (Po indicato come dentale, LIA e Pn nella categoria sbagliata). | **Corretto** (`CATEGORIE` in `evaluate_refine.py`). |
| 10 | **Il vincolo geometrico sull'incisivo superiore era invertito** e imponeva la punta (UIT) sopra l'apice (UIA). | **Corretto**: la punta deve stare sotto l'apice. |
| 11 | **Griglia pixel incoerente.** I target erano disegnati in `x·256` ma il soft-argmax decodificava in `j/255`: un errore sistematico fino a circa 0.9 mm ai bordi. | **Corretto** con la convenzione `"centri"` per i nuovi training. I checkpoint vecchi vengono decodificati con `"legacy"`, come in origine. |
| 12 | La media Senior/Junior di Aariz usava `np.ceil` (bias fino a +1 px). | **Corretto** (media semplice). |
| 13 | Messaggi che citavano `analyze_landmark_difficulty.py` e script in `Test/` non aggiornati. | **Corretto** (resta solo `test_ceph.py`, vedi §13). |
| 14 | La validazione è un 20% casuale, non il test split ufficiale: i risultati non sono confrontabili con i benchmark pubblici. | Non modificato (è una scelta di design). |

**Nota sui checkpoint esistenti.** Le correzioni 1, 2, 8 e 9 cambiano la valutazione
anche dei modelli già addestrati: basta rieseguire `evaluate_refine.py`.
Le correzioni 4, 5, 6, 10, 11 e 12 riguardano il training e hanno effetto solo dopo
un nuovo addestramento.

La correzione 5 cambia anche le etichette di validazione delle immagini ISBI. Per
questo **i risultati di `model_no_unife/results.txt` non sono confrontabili** con
quelli nuovi: il vecchio modello è stato addestrato e valutato con etichette ISBI
sbagliate, ma coerenti tra loro, e può aver imparato a riprodurle.

---

## 17. Sigma addestrabile e incertezza

La modalità si sceglie con `CONFIG["heatmap_mode"]` in `train_heatmap.py`. Viene
salvata nel checkpoint, quindi valutazione e inferenza la riconoscono da sole.

| Modalità | Target / loss | Cosa si impara | Incertezza per immagine |
|---|---|---|---|
| `"fixed"` (default, come prima) | Gaussiana con σ fisso + Adaptive Wing | — | Dispersione della heatmap attorno al picco. Include sempre la larghezza del target (circa 4 px), quindi va calibrata. |
| `"learned_sigma"` | Gaussiana ellittica con σx, σy, ρ **addestrabili per landmark** + MSE + penalità sull'area | Quanto ogni landmark è intrinsecamente ambiguo (Payer et al., 2020) | Gaussiana adattata alla heatmap predetta |
| `"probabilistic"` | Heatmap = logit di una distribuzione spaziale (softmax) + NLL della posizione vera | Una distribuzione larga quanto l'errore reale del modello | Covarianza della distribuzione predetta |

### Sigma addestrabile (`SigmaAddestrabile`)
- Ogni landmark ha tre parametri: σx e σy (tra `sigma_min` e `sigma_max`, in pixel
  della heatmap) e ρ (|ρ| < 0.9).
- I valori iniziali vengono da `SIGMA_PER_LANDMARK`.
- Il target è una gaussiana *normalizzata* moltiplicata per γ = 2π·4². Con σ = 4 il
  picco vale 1; con σ più grande il picco si abbassa, quindi allargare σ "costa".
- La penalità `sigma_penalita · area/4²` (default 0.1) impedisce che σ cresca
  all'infinito. Il valore va tarato: se le σ convergono tutte a `sigma_max`, la
  penalità è troppo bassa; se convergono tutte a `sigma_min`, è troppo alta.
- Le σ hanno un learning rate proprio (`lr_sigma`, 1e-3). Ogni 10 epoche il log ne
  riporta la media; a fine training stampa i valori per landmark, in pixel e in mm
  indicativi.

### Heatmap probabilistica
- Il canale heatmap è un logit. La loss (`nll_heatmap_loss`) è la negative
  log-likelihood della posizione vera, interpolata bilinearmente, divisa per
  log(H·W).
- La coordinata è il valore atteso della distribuzione (soft-argmax con temperatura 1).

### Fonti di incertezza
Tutte espresse in mm (RMS dell'errore radiale):

- `heatmap`: dalla covarianza della heatmap (`momenti_heatmap`);
- `tta`: dalla dispersione delle 8 predizioni TTA (`covarianza_tta`);
- `totale`: √(heatmap² + tta²).

### Calibrazione (`evaluate_refine.py`)
Le dispersioni grezze non sono probabilità tarate. `analisi_incertezza` misura per
ogni fonte:

- la **correlazione di Spearman** tra incertezza ed errore (un'incertezza utile ha
  correlazione chiaramente positiva);
- l'**errore medio per quintile** di incertezza (dovrebbe crescere);
- un **fattore conforme q** per landmark, tale che il raggio `q·σ` contenga il punto
  vero con probabilità ≥ `copertura` (default 90%). q viene stimato sulle immagini
  pari e verificato sulle dispari, poi ristimato su tutte e salvato in
  `calibrazione_incertezza.json`.

`predict_heatmap.py` legge questo file e mostra per ogni landmark σ e il raggio r90.
Sull'immagine disegna un cerchio di raggio r90 e salva `<immagine>_pred.png`.
La calibrazione vale solo per il checkpoint, la modalità e la TTA con cui è stata
calcolata: va rifatta dopo ogni training.

### Come confrontare le modalità
1. Addestrare Stage 1 (e 2) con ciascuna `heatmap_mode`, cambiando `save_path`.
2. Eseguire `evaluate_refine.py` per ciascun checkpoint.
3. Confrontare MRE/SDR (la precisione non deve peggiorare) e la sezione
   UNCERTAINTY: Spearman più alto, errore crescente per quintile, copertura vicina
   al 90% con il raggio medio più piccolo.
