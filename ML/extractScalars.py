from tensorboard.backend.event_processing import event_accumulator
from glob import glob
import numpy as np
import sys

extract='epoch_val_classification_acc'

runs=glob("log_dir/xval_smooth_combined_rw500_fold*_step*")
print(runs)
firstr=True
collected=[]
steps=[]
for run in runs:
    ea=event_accumulator.EventAccumulator(path=run)
    ea.Reload()
    try:
        scs=ea.Scalars(extract)
    except Exception:
        print("Skipping")
        continue
    da=[]
    sa=[]
    for s in scs:
        sa.append(s.step)
        da.append(s.value)
    da=np.array(da)
    if da.shape[0]<250:
        print("skipping")
        continue
    da=np.expand_dims(da, axis=1)
    if firstr:
        collected=np.array(da)
        print(da.shape)
        firstr=False
        steps=np.expand_dims(sa, axis=1)
    else:
#        print(collected.shape)
        if da.shape[0] <250:
#            print(da.shape)
            continue
        collected=np.concatenate((collected, da),axis=1)
print(f"collected {collected.shape}")
avgs=np.expand_dims(collected.mean(axis=1),axis=1)
stds=np.expand_dims(collected.std(axis=1),axis=1)
#print(avgs.shape)
#print(stds.shape)
res=np.concatenate((steps, avgs, stds), axis=1)
#print(res)
#print(collected)
#print(collected.shape)
np.savetxt(sys.stdout, res)
