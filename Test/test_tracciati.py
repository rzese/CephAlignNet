from src.datasets import Datasets

d = Datasets(tracciati_path='data/tracciati_forniti')
img, pts, mask, bbox, ps = d[d.len_aariz + d.len_isbi]

print('Primo punto Tracciati:', pts[0].tolist())
print('Max coordinata:', pts[mask==1].max().item())
print('Min coordinata (non zero):', pts[mask==1].min().item())
print()
print('Tutti i punti validi:')
from src.dataset import SYMBOL_ORDER as NOMI
print('Pixel size:', ps.item(), 'mm/px')
for i in range(len(NOMI)):
    if mask[i] == 1:
        print(f'  {NOMI[i]:5s}: x={pts[i,0]:.1f}  y={pts[i,1]:.1f}')