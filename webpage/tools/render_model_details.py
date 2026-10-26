"""Render website-only model illustrations from publication sources.

Run from any directory with the repository's Python environment. Reads the
bundled Supplementary Data and existing figure definitions; does not run OPF
or modify publication artwork. Supply curves are capped at unity for this
availability illustration, unlike the raw empirical curves in the SI.
"""

from pathlib import Path
import sys
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'vis/figure_source'
sys.path.insert(0, str(SOURCE / 'code'))
sys.path.insert(0, str(ROOT))
from demand_calibration_definition import model_curve
from generator_derating_definition import formula_values, displayed_mask, temperature_values
from conductor_heat_flow_definition import MODEL_SPECS, WEATHER, evaluate_grid
from utils.heat_flow_utils import heat_banlance_equation, maximum_allowable_current

OUT = ROOT / 'webpage/images/models'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                     'axes.labelsize': 13, 'axes.labelweight': 'bold',
                     'axes.spines.top': False, 'axes.spines.right': False})

def axes():
    fig, ax = plt.subplots(figsize=(6.6, 4.6), dpi=200)
    fig.subplots_adjust(left=.14, right=.97, bottom=.16, top=.84)
    ax.grid(alpha=.15)
    ax.set_axisbelow(True)
    return fig, ax

def save(fig, name):
    fig.savefig(OUT / f'{name}.png', facecolor='white', dpi=200)
    plt.close(fig)

with zipfile.ZipFile(SOURCE / 'data/Supplementary_Data_01_Figs01-08.zip') as archive:
    base = 'data/Supplementary_Fig_06_demand_calibration/data/'
    observations = pd.read_csv(archive.open(base + 'demand_calibration_plotted_daily_observations.csv'))
    parameters = pd.read_csv(archive.open(base + 'demand_calibration_model_parameters.csv'))
data = observations[observations.country_code == 'ES']
params = parameters[parameters.country_code == 'ES'].iloc[0].to_dict()
x = np.linspace(data.population_weighted_bait_c.min(), data.population_weighted_bait_c.max(), 1000)
fig, ax = axes()
for weekday, label, colour, marker in [(True, 'Weekday', '#b43a32', 'o'), (False, 'Weekend / holiday', '#2d5f9a', 's')]:
    subset = data[(data.day_type == 'weekday') == weekday]
    ax.scatter(subset.population_weighted_bait_c, subset.observed_daily_demand_gw,
               s=7, color=colour, alpha=.22, marker=marker, linewidths=0)
    ax.plot(x, model_curve(x, params, weekday=weekday), color=colour, lw=2.3, label=label)
ax.set(xlabel='BAIT (°C)', ylabel='Daily demand (GW)')
ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.03), ncol=2, frameon=False, fontsize=10)
save(fig, 'demand')

fig, ax = axes()
x = temperature_values()
for key, label, colour, style in [('ocgt', 'OCGT', '#2d5f9a', '-'), ('ccgt', 'CCGT', '#b66a13', '--'),
                                  ('nuclear', 'Nuclear', '#246d78', '-.'), ('copper_windings', 'Copper windings', '#b43a32', ':')]:
    mask = displayed_mask(key, x)
    ax.plot(x[mask], np.minimum(formula_values(x)[key][mask], 1), label=label, color=colour, ls=style, lw=2.3)
ax.set(xlabel='Air temperature (°C)', ylabel='Derating factor', xlim=(10, 50), ylim=(.7, 1.03))
ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=2, frameon=False, fontsize=10)
save(fig, 'supply')

fig, ax = axes()
fig.subplots_adjust(right=.80)
conductor = dict(MODEL_SPECS[1]['conductor'])
air, current, temperature = evaluate_grid(heat_banlance_equation, conductor)
filled = ax.contourf(air, current, temperature, levels=np.arange(45, 141, 5), cmap='coolwarm', extend='both')
limits = []
for value in air:
    weather = dict(WEATHER, air_temperature=np.array([value]))
    limits.append(float(np.asarray(maximum_allowable_current(conductor, weather)).reshape(-1)[0]) * conductor['num_bundle'])
ax.plot(air, limits, color='#b43a32', lw=2.3, label='90 °C thermal limit')
ax.axhline(2580, color='#2d5f9a', ls='--', lw=2.3, label='Nominal current')
ax.set(xlabel='Air temperature (°C)', ylabel='Bundle current (A)', xlim=(25,50), ylim=(current.min(),current.max()))
ax.legend(loc='lower center', bbox_to_anchor=(.5,1.01), ncol=1, frameon=False, fontsize=10)
cax = fig.add_axes([.83, .16, .025, .68])
fig.colorbar(filled, cax=cax, label='Conductor temperature (°C)')
save(fig, 'transmission')
print('Rendered demand, supply and transmission website illustrations.')
