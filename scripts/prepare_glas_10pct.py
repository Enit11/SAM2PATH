"""Create one CSV for official GlaS train/test with 10% labelled training."""
from pathlib import Path
import argparse, random
import pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='Glas/GlasDataset'); ap.add_argument('--out',default='dataset_cfg/glas_10pct.csv'); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--val-ratio',type=float,default=.15)
    a=ap.parse_args(); root=Path(a.root); names=sorted(p.stem for p in root.glob('train_*.bmp') if not p.stem.endswith('_anno'))
    tests=sorted(p.stem for p in root.glob('testA_*.bmp') if not p.stem.endswith('_anno'))+sorted(p.stem for p in root.glob('testB_*.bmp') if not p.stem.endswith('_anno'))
    rng=random.Random(a.seed); shuffled=names[:]; rng.shuffle(shuffled)
    nval=max(1,round(len(names)*a.val_ratio)); val=set(shuffled[:nval]); remain=[x for x in shuffled if x not in val]
    nl=max(1,round(len(names)*.10)); labeled=set(remain[:nl])
    rows=[]
    for n in names: rows.append({'img_id':n,'is_test':2 if n in val else 0,'labeled':int(n in labeled)})
    for n in tests: rows.append({'img_id':n,'is_test':1,'labeled':1})
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(a.out,index=False)
    print(f'train={len(names)} val={len(val)} labeled={len(labeled)} unlabeled={len(names)-len(val)-len(labeled)} test={len(tests)} -> {a.out}')
if __name__=='__main__': main()
