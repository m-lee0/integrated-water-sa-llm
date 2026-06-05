from pathlib import Path
import pandas as pd
from SALib.analyze import sobol as sobol_analyze
from experimentor import formulate_salib_problem, SAMPLE_N
from parameters.sa_parameters_bwick import params_to_select

# Load and concatenate all per-iteration result files
results_fid = Path(__file__).parent.parent / 'SA' / 'results'
all_files = sorted(results_fid.glob('iter_*.csv'))

if not all_files:
    raise FileNotFoundError(f"No result files found in {results_fid}")

df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
df = df.sort_values(by='iter').reset_index(drop=True)

print(f"Loaded {len(df)} model runs")

Y = df['mean_flow'].values

problem = formulate_salib_problem(params_to_select())
del problem['groups']

Si = sobol_analyze.analyze(problem, Y, calc_second_order=True, seed=1, print_to_console=True)

names = problem['names']      # ['surface_coefficient', 'percolation_coefficient']
k = problem['num_vars']       # 2
N = SAMPLE_N

lines = []
lines.append(f"k={k} parameters and N={N} samples.")
lines.append("")
lines.append("ST    ST_conf")
for name, st, st_c in zip(names, Si['ST'], Si['ST_conf']):
    lines.append(f"{name}    {st:.6f}    {st_c:.6f}")
lines.append("")
lines.append("S1    S1_conf")
for name, s1, s1_c in zip(names, Si['S1'], Si['S1_conf']):
    lines.append(f"{name}    {s1:.6f}    {s1_c:.6f}")
lines.append("")
lines.append("S2    S2_conf")
for i in range(k):
    for j in range(i + 1, k):
        lines.append(f"[{names[i]}, {names[j]}]    {Si['S2'][i][j]:.6f}    {Si['S2_conf'][i][j]:.6f}")

formatted = "\n".join(lines)

output_path = results_fid / 'sa_results_formatted.txt'
output_path.write_text(formatted)
print(f"Saved to {output_path}")