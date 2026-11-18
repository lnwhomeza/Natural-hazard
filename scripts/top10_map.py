"""Map of the 10 fastest-sinking districts (median vertical rate, results/risk/risk_districts.csv h_rate).

    python scripts/top10_map.py docs/slides/map_top10.png
"""
import sys
import geopandas as gpd
import matplotlib
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402

plt.rcParams['font.family'] = ['Loma', 'DejaVu Sans']
adm = gpd.read_file('boundaries/districts_bmr.gpkg').to_crs(32647)
r = pd.read_csv('results/risk/risk_districts.csv')[['adm2_pcode', 'h_rate']]
top = r.sort_values('h_rate', ascending=False).head(10).reset_index(drop=True)
top['rank'] = top.index + 1
d = adm.merge(top, on='adm2_pcode', how='left')
col = d['rank'].map(lambda k: '#5c6898' if k <= 3 else ('#c9bfd9' if k <= 10 else '#f1ece3'))
fig, ax = plt.subplots(figsize=(10, 6.2), dpi=220)
d.plot(ax=ax, color=col, edgecolor='#ffffff', linewidth=.6)
d.dissolve('adm1_name').boundary.plot(ax=ax, color='#7a6f63', lw=.9)
for _, row in d[d['rank'].notna()].iterrows():
    p = row.geometry.representative_point()
    k = int(row['rank'])
    ax.text(p.x, p.y, str(k), ha='center', va='center', fontsize=15, fontweight='bold',
            color='white' if k <= 3 else '#343434',
            path_effects=[pe.withStroke(linewidth=2.5, foreground='#5c6898' if k <= 3 else 'white')])
ax.set_axis_off()
fig.savefig(sys.argv[1], bbox_inches='tight', pad_inches=.05, facecolor='white')
print(top.merge(adm[['adm2_pcode', 'adm2_name1']], on='adm2_pcode')[['rank', 'adm2_name1', 'h_rate']].to_string(index=False))
