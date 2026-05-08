#!/usr/bin/env python3
import json,glob,csv,os

# helper to read roofline.json
def read_roofline_json(path):
    try:
        with open(path) as f:
            j=json.load(f)
        gflops=None
        dram_mb=None
        try:
            gflops=j['empirical']['gflops']['data'][0][1]
        except Exception:
            gflops=None
        try:
            dram_mb=j['empirical']['gbytes']['data'][0][1]
        except Exception:
            dram_mb=None
        return float(gflops) if gflops is not None else None, float(dram_mb) if dram_mb is not None else None
    except Exception:
        return None,None

# helper to parse sum file for FP64 and DRAM
def read_sum_file(path):
    gflops=None
    dram=None
    with open(path) as f:
        for line in f:
            l=line.strip()
            if l.endswith('FP64 GFLOPs'):
                try:
                    gflops=float(l.split()[0])
                except:
                    pass
            if l.endswith('DRAM'):
                try:
                    dram=float(l.split()[0])
                except:
                    pass
    return gflops, dram

# get max achieved GFLOPS from a benchmark CSV
def max_gflops_from_csv(path):
    try:
        mx=0.0
        with open(path) as f:
            r=csv.DictReader(f)
            for row in r:
                try:
                    v=float(row.get('gflops',0))
                    if v>mx: mx=v
                except:
                    pass
        return mx
    except Exception:
        return None

# locate runs
my_json='roofline/mylaptop/Results/Run.002/roofline.json'
my_json_alt='roofline/mylaptop/Results/Run.001/roofline.json'
friend_sum_candidates=glob.glob('roofline/Results/Run.001/FLOPS.001/OpenMP.*/sum')

my_gflops,my_dram_mb=None,None
for p in [my_json,my_json_alt]:
    if os.path.exists(p):
        my_gflops,my_dram_mb=read_roofline_json(p)
        break

friend_gflops,friend_dram_mb=None,None
if friend_sum_candidates:
    # pick sum with largest FP64 if multiple
    best=None
    best_val=-1
    for s in friend_sum_candidates:
        g,d=read_sum_file(s)
        if g is not None and g>best_val:
            best_val=g; best=s
    if best:
        friend_gflops,friend_dram_mb=read_sum_file(best)

# benchmark CSVs
my_bench='benchmarks/my_laptop/benchmark_results.csv'
friend_bench='benchmarks/benchmark_results.csv'
my_max = max_gflops_from_csv(my_bench)
friend_max = max_gflops_from_csv(friend_bench)

# compute GB/s
my_dram_gb = (my_dram_mb/1024.0) if my_dram_mb else None
friend_dram_gb = (friend_dram_mb/1024.0) if friend_dram_mb else None

# basic conclusions
lines=[]
lines.append('Comparison of ERT measurements and FFT benchmark results')
lines.append('')
if my_gflops is not None:
    lines.append(f"My laptop: Peak GFLOP/s (ERT) = {my_gflops:.2f} GFLOP/s; DRAM = {my_dram_mb:.2f} MB/s ({my_dram_gb:.2f} GB/s)")
else:
    lines.append('My laptop: no ERT roofline.json found')
if my_max is not None:
    lines.append(f"My laptop: best FFT kernel observed = {my_max:.2f} GFLOP/s")

lines.append('')
if friend_gflops is not None:
    lines.append(f"Friend's laptop: Peak GFLOP/s (ERT) = {friend_gflops:.2f} GFLOP/s; DRAM = {friend_dram_mb:.2f} MB/s ({friend_dram_gb:.2f} GB/s)")
else:
    lines.append("Friend's laptop: no ERT summary found")
if friend_max is not None:
    lines.append(f"Friend's laptop: best FFT kernel observed = {friend_max:.2f} GFLOP/s")

lines.append('')
# simple interpretation
if my_gflops and my_max is not None:
    frac = my_max / my_gflops
    if frac > 0.5:
        lines.append('My laptop analysis: kernel is likely compute-bound (achieved >50% of peak).')
    else:
        lines.append('My laptop analysis: kernel is memory-bound or limited by data-locality/synchronization (achieved << peak).')
if friend_gflops and friend_max is not None:
    frac = friend_max / friend_gflops
    if frac > 0.5:
        lines.append("Friend's laptop analysis: kernel is likely compute-bound (achieved >50% of peak).")
    else:
        lines.append("Friend's laptop analysis: kernel is memory-bound or limited by data-locality/synchronization (achieved << peak).")

# write summary
out_dir='roofline/results'
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir,'comparison.txt'),'w') as f:
    f.write('\n'.join(lines))
print('\n'.join(lines))
print('\nWrote summary to roofline/results/comparison.txt')
