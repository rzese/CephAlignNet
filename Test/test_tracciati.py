from src.datasets import Datasets

d = Datasets(tracciati_path='data/tracciati_forniti')
img, pts, mask, bbox, ps = d[d.len_aariz + d.len_isbi + d.len_ceph]

print('Primo punto Tracciati:', pts[0].tolist())
print('Max coordinata:', pts[mask==1].max().item())
print('Min coordinata (non zero):', pts[mask==1].min().item())
print()
print('Tutti i punti validi:')
NOMI = ["A","ANS","B","Me","N","Or","Pog","PNS","R","S",
        "Ar","Co","Gn","Go","Po","LPM","LIT","LMT","UPM"]
for i in range(19):
    if mask[i] == 1:
        print(f'  {NOMI[i]:5s}: x={pts[i,0]:.1f}  y={pts[i,1]:.1f}')