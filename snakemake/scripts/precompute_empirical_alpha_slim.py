"""
Compute empirical alpha (true proportion of beneficial substitutions) from SLiM by directly
counting fixed beneficial vs deleterious substitutions, for the s_b=1e-3 dominance family across
dominance coefficients h. Several replicates are pooled to reduce noise. The result is cached so the
divergence machinery can be validated against a real ground truth (not an analytical proxy).

Run from the repo root. Requires SLiM on PATH (conda-forge `slim`).
"""
import json, os, re, subprocess
from concurrent.futures import ThreadPoolExecutor

SLIM = 'snakemake/scripts/count_substitutions.slim'  # minimal substitution-counting model
FAMILY = dict(s_b=1e-3, b=0.3, s_d=3e-2, p_b=0.05)
N, mu, r, L, G = 1000, 1e-8, 1e-7, 1e7, 10000
H_VALUES = [round(0.1 * i, 1) for i in range(0, 11)]  # 0.0 .. 1.0
N_REP = 3
OUT = 'testing/cache/slim/empirical_alpha/s_b=1e-3_b=0.3_s_d=3e-2_p_b=0.05.json'

def run(h, seed):
    cmd = ['slim', '-s', str(seed),
           '-d', f'mu={mu}', '-d', f's_d={FAMILY["s_d"]}', '-d', f's_b={FAMILY["s_b"]}',
           '-d', f'b={FAMILY["b"]}', '-d', f'p_b={FAMILY["p_b"]}', '-d', f'h={h}',
           '-d', f'N={N}', '-d', f'r={r}', '-d', f'L={L}', '-d', f'G={G}', SLIM]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    m = re.search(r'EMPIRICAL_RESULT del=(\d+) ben=(\d+)', out)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

def empirical_alpha(h):
    dele = bene = 0
    for seed in range(1, N_REP + 1):
        d, b = run(h, seed)
        dele += d; bene += b
    return dict(h=h, n_del=dele, n_ben=bene, alpha=bene / (dele + bene) if (dele + bene) else float('nan'))

with ThreadPoolExecutor(max_workers=8) as ex:
    results = list(ex.map(empirical_alpha, H_VALUES))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
data = dict(family=FAMILY, N=N, mu=mu, r=r, L=L, G=G, n_replicates=N_REP,
            alpha={str(rdict['h']): rdict for rdict in results})
json.dump(data, open(OUT, 'w'), indent=2)
for rdict in results:
    print(f"CACHED h={rdict['h']}: del={rdict['n_del']} ben={rdict['n_ben']} alpha={rdict['alpha']:.3f}", flush=True)
