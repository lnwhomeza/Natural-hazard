"""Study-area map: 59 districts of Bangkok, Samut Prakan and Samut Sakhon,
coloured by province, with the two GNSS stations used for validation.

Output: results/study_area.png
"""
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

for f in ['/usr/share/fonts/opentype/tlwg/Loma.otf', '/usr/share/fonts/opentype/tlwg/Loma-Bold.otf']:
    font_manager.fontManager.addfont(f)
plt.rcParams.update({'font.family': 'Loma', 'axes.unicode_minus': False})

STATIONS = {'CUSV': (13.7359, 100.5339), 'KMIT': (13.7278, 100.7724)}  # same as full_check.py
PROV = {  # adm1_name -> (Thai label, colour)
    'Bangkok': ('กรุงเทพมหานคร', '#9fb3e0'),
    'Samut Prakan': ('สมุทรปราการ', '#c9d5f0'),
    'Samut Sakhon': ('สมุทรสาคร', '#e3e9f7'),
}
NAVY = '#1f3354'

g = gpd.read_file('boundaries/districts_bmr.gpkg').to_crs(32647)
prov = g.dissolve('adm1_name').reset_index()
area = g.area.sum() / 1e6

fig, ax = plt.subplots(figsize=(11, 8.2), dpi=200)
fig.patch.set_facecolor('white')
for name, (th, col) in PROV.items():
    g[g.adm1_name == name].plot(ax=ax, color=col, edgecolor='white', linewidth=0.6)
prov.boundary.plot(ax=ax, color=NAVY, linewidth=1.6)

for name, (th, _) in PROV.items():
    p = prov[prov.adm1_name == name].geometry.iloc[0].representative_point()
    dy = 14000 if name == 'Bangkok' else 0  # keep the label clear of the CUSV marker
    ax.annotate(th, (p.x, p.y + dy), ha='center', va='center', fontsize=17, fontweight='bold', color=NAVY)

st = gpd.GeoDataFrame(geometry=gpd.points_from_xy([v[1] for v in STATIONS.values()],
                                                  [v[0] for v in STATIONS.values()]), crs=4326).to_crs(32647)
ax.scatter(st.geometry.x, st.geometry.y, marker='^', s=150, color='#d94a45', edgecolor='white', zorder=5)
for (k, _), pt in zip(STATIONS.items(), st.geometry):
    ax.annotate(k, (pt.x, pt.y), xytext=(8, 8), textcoords='offset points', fontsize=12,
                fontweight='bold', color='#d94a45')

# room below the coast for the gulf label and scale bar
ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin - 9000, ymax)
xmid = sum(ax.get_xlim()) / 2
ax.text(xmid, ymin - 2500, 'อ่าวไทย', ha='center', va='center', fontsize=14, color='#5b6f93', style='italic')
# scale bar (20 km) and north arrow
x0, y0 = ax.get_xlim()[0] + 4000, ymin - 7000
ax.plot([x0, x0 + 20000], [y0, y0], color=NAVY, linewidth=4, solid_capstyle='butt')
ax.text(x0 + 10000, y0 + 1500, '20 กม.', ha='center', va='bottom', fontsize=12, color=NAVY)
xn, yn = ax.get_xlim()[1] - 6000, ax.get_ylim()[1] - 12000
ax.annotate('N', xy=(xn, yn + 9000), xytext=(xn, yn), ha='center', fontsize=15, fontweight='bold',
            color=NAVY, arrowprops=dict(arrowstyle='-|>', color=NAVY, lw=2))

handles = [Patch(facecolor=c, edgecolor=NAVY, label=t) for t, c in PROV.values()]
handles.append(Line2D([], [], marker='^', linestyle='', color='#d94a45', markersize=11, label='สถานี GNSS'))
ax.legend(handles=handles, loc='lower right', frameon=True, fontsize=12, framealpha=0.95)
ax.set_title(f'พื้นที่ศึกษา: 59 เขต/อำเภอ  ({area:,.0f} ตร.กม.)', fontsize=16, color=NAVY, pad=10)
ax.set_axis_off()
fig.tight_layout()
fig.savefig('results/study_area.png', facecolor='white')
print(f'area {area:.0f} km2')
