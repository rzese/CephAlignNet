import sys
import os
# Importiamo la classe dal tuo file modificato
from src.dataset_tracciati import TracciatiDataset, _identifica_file, _identifica_file, _parse_excel

def testare_dataset(base_path="data/tracciati_forniti"):
    print("=" * 70)
    print(f" AVVIO TEST DATASET SU: {base_path}")
    print("=" * 70)

    if not os.path.isdir(base_path):
        print(f"ERRORE: La cartella '{base_path}' non esiste sul cluster!")
        return

    # Inizializziamo il dataset (questo stamperà già il riepilogo iniziale)
    print("\n--- Inizializzazione Dataset ---")
    dataset = TracciatiDataset(base_path=base_path)
    print("-" * 40)

    # Scansione manuale di tutte le sotto-cartelle per vedere i dettagli precisi
    subfolders = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and not d.startswith(".")])
    
    totale_pazienti = 0
    letti_con_successo = 0

    for sub in subfolders:
        sub_path = os.path.join(base_path, sub)
        print(f"\n📁 Cartella Principale: {sub}")
        
        case_ids = sorted([d for d in os.listdir(sub_path) if os.path.isdir(os.path.join(sub_path, d)) and not d.startswith(".")])
        
        for cid in case_ids:
            totale_pazienti += 1
            case_path = os.path.join(sub_path, cid)
            img_path, data_path = _identifica_file(case_path, cid)
            
            if data_path is None:
                print(f"  ❌ [Paziente {cid}]: FILE DATI NON TROVATO!")
                continue
                
            # Capiamo l'estensione del file trovato
            estensione = os.path.splitext(data_path)[1].lower()
            
            # Proviamo a fare il parsing dei landmark
            try:
                # Simuliamo un'altezza immagine fissa (es. 2000) solo per il test del flip_y
                landmarks = _parse_excel(data_path, coord_scale=1.0, flip_y=False, img_height=2000)
                num_lm = len(landmarks)
                
                if num_lm > 0:
                    print(f"  ✅ [Paziente {cid}]: Letto file {estensione} ({num_lm} landmark trovati)")
                    letti_con_successo += 1
                else:
                    print(f"  ⚠️ [Paziente {cid}]: Trovato file {estensione} MA NESSUN LANDMARK RICONOSCIUTO (controlla NAME_MAP)")
            except Exception as e:
                print(f"  ❌ [Paziente {cid}]: Errore durante la lettura del file {estensione}: {e}")

    print("\n" + "=" * 70)
    print(" BILANCIO FINALE DEL TEST")
    print("=" * 70)
    print(f"Pazienti totali scansionati: {totale_pazienti}")
    print(f"Pazienti letti con successo:  {letti_con_successo}/{totale_pazienti}")
    if totale_pazienti == letti_con_successo:
        print("🎉 OTTIMO! Tutti i dati vengono letti correttamente senza skippare nulla!")
    else:
        print("❌ Ci sono ancora dei problemi su alcuni file. Controlla i log qui sopra.")
    print("=" * 70)

if __name__ == "__main__":
    # Se passi un percorso da terminale usa quello, altrimenti usa il default
    percorso = sys.argv[1] if len(sys.argv) > 1 else "data/tracciati_forniti"
    testare_dataset(percorso)