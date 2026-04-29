import pandas as pd
import pypsa
import matplotlib.pyplot as plt

# ── Load data ──────────────────────────────────────────────────────────────────
df_elec = pd.read_csv('data/electricity_demand.csv', sep=';', index_col=0)
df_elec.index = pd.to_datetime(df_elec.index)

df_onshorewind = pd.read_csv('data/onshore_wind_1979-2017.csv', sep=';', index_col=0)
df_onshorewind.index = pd.to_datetime(df_onshorewind.index)

df_offshorewind = pd.read_csv('data/offshore_wind_1979-2017.csv', sep=';', index_col=0)
df_offshorewind.index = pd.to_datetime(df_offshorewind.index)

df_solar = pd.read_csv('data/pv_optimal.csv', sep=';', index_col=0)
df_solar.index = pd.to_datetime(df_solar.index)

country = 'DNK'

# ── Helper ─────────────────────────────────────────────────────────────────────
def annuity(n, r):
    if r > 0:
        return r / (1. - 1. / (1. + r) ** n)
    else:
        return 1 / n

# ── Build network ──────────────────────────────────────────────────────────────
network = pypsa.Network()
hours_in_2015 = pd.date_range('2015-01-01 00:00Z', '2015-12-31 23:00Z', freq='h')
network.set_snapshots(hours_in_2015.values)
network.add("Bus", "electricity bus")
network.add("Load", "load", bus="electricity bus", p_set=df_elec[country].values)

network.add("Carrier", "gas", co2_emissions=0.19)
network.add("Carrier", "onshorewind")
network.add("Carrier", "offshorewind")
network.add("Carrier", "solar")

CF_wind = df_onshorewind[country][[h.strftime("%Y-%m-%dT%H:%M:%SZ") for h in network.snapshots]]
network.add("Generator", "Onshore wind", bus="electricity bus",
            p_nom_extendable=True, carrier="onshorewind",
            capital_cost=annuity(27, 0.07) * 1118775, marginal_cost=0,
            p_max_pu=CF_wind.values)

CF_wind_off = df_offshorewind[country][[h.strftime("%Y-%m-%dT%H:%M:%SZ") for h in network.snapshots]]
network.add("Generator", "Offshore wind", bus="electricity bus",
            p_nom_extendable=True, carrier="offshorewind",
            capital_cost=annuity(27, 0.07) * 2115944, marginal_cost=0,
            p_max_pu=CF_wind_off.values)

CF_solar = df_solar[country][[h.strftime("%Y-%m-%dT%H:%M:%SZ") for h in network.snapshots]]
network.add("Generator", "Solar", bus="electricity bus",
            p_nom_extendable=True, carrier="solar",
            capital_cost=annuity(25, 0.07) * 450000, marginal_cost=0,
            p_max_pu=CF_solar.values)

network.add("Generator", "OCGT", bus="electricity bus",
            p_nom_extendable=True, carrier="gas",
            capital_cost=annuity(25, 0.07) * 453960,
            marginal_cost=30 / 0.41)

network.add("Generator", "CCGT", bus="electricity bus",
            p_nom_extendable=True, carrier="gas",
            capital_cost=annuity(25, 0.07) * 880000,
            marginal_cost=30 / 0.56)

# ── Optimise ───────────────────────────────────────────────────────────────────
network.optimize(solver_name='gurobi', solver_options={"OutputFlag": 0, "LogToConsole": 0})

print(f"Total system cost: {network.objective / 1e6:.2f} M€/yr")
print(f"Average cost:      {network.objective / network.loads_t.p.sum().sum():.2f} €/MWh")
print("\nOptimal capacities [GW]:")
print(network.generators.p_nom_opt.div(1e3))
print("\nAnnual generation [TWh]:")
print(network.generators_t.p.sum().div(1e6))

# ── Plotting ───────────────────────────────────────────────────────────────────
labels      = ['Onshore wind', 'Offshore wind', 'Solar PV', 'Gas (OCGT)', 'Gas (CCGT)']
model_names = ['Onshore wind', 'Offshore wind', 'Solar',    'OCGT',       'CCGT']
colors      = ['blue', 'dodgerblue', 'orange', 'crimson', 'darkviolet']

# Drop technologies with zero optimal capacity
active = [(l, n, c) for l, n, c in zip(labels, model_names, colors)
          if network.generators.p_nom_opt[n] > 0]
labels_a, model_names_a, colors_a = zip(*active) if active else (labels, model_names, colors)

# Installed capacity pie chart
cap_sizes = [network.generators.p_nom_opt[n] for n in model_names_a]
plt.figure(figsize=(6, 5), dpi=300)
plt.pie(cap_sizes, colors=colors_a,
        labels=[f'{l}\n{s/1e3:.1f} GW' for l, s in zip(labels_a, cap_sizes)],
        wedgeprops={'linewidth': 0})
plt.axis('equal')
plt.title('Installed capacity mix', y=1.05, fontweight='bold')
plt.tight_layout()
plt.savefig('1a_installed_capacity_mix.png', dpi=300)
plt.show()

# Dispatch: first week of January (winter)
plt.figure(figsize=(8, 5), dpi=300)
plt.plot(network.loads_t.p['load'][0:168], color='grey', label='Demand', lw=4)
for n, l, c in zip(model_names_a, labels_a, colors_a):
    plt.plot(network.generators_t.p[n][0:168], color=c, label=l)
plt.ylabel('Generation [MWh/h]')
plt.title('Electricity generation — first week of January 2015')
plt.xlabel('Time')
plt.xlim(network.snapshots[0], network.snapshots[167])
plt.xticks(rotation=45)
plt.legend(fancybox=True, shadow=True, loc='best')
plt.tight_layout()
plt.savefig('1a_timeseries_winter.png', dpi=300)
plt.show()

# Dispatch: first week of July (summer)
plt.figure(figsize=(8, 5), dpi=300)
plt.plot(network.loads_t.p['load'][4344:4512], color='grey', label='Demand', lw=4)
for n, l, c in zip(model_names_a, labels_a, colors_a):
    plt.plot(network.generators_t.p[n][4344:4512], color=c, label=l)
plt.ylabel('Generation [MWh/h]')
plt.title('Electricity generation — first week of July 2015')
plt.xlabel('Time')
plt.xlim(network.snapshots[4344], network.snapshots[4511])
plt.xticks(rotation=45)
plt.legend(fancybox=True, shadow=True, loc='best')
plt.tight_layout()
plt.savefig('1a_timeseries_summer.png', dpi=300)
plt.show()

# Annual generation mix pie chart
sizes = [network.generators_t.p[n].sum() for n in model_names_a]
percentages = [s / sum(sizes) * 100 for s in sizes]
plt.figure(figsize=(6, 5), dpi=300)
plt.pie(sizes, colors=colors_a,
        labels=[f'{l}\n{p:.1f}%' for l, p in zip(labels_a, percentages)],
        wedgeprops={'linewidth': 0})
plt.axis('equal')
plt.title('Annual electricity generation mix', y=1.05, fontweight='bold')
plt.tight_layout()
plt.savefig('1a_annual_generation_mix.png', dpi=300)
plt.show()

# Duration curves
plt.figure(figsize=(8, 5), dpi=300)
for n, l, c in zip(model_names_a, labels_a, colors_a):
    sorted_gen = network.generators_t.p[n].sort_values(ascending=False).reset_index(drop=True)
    plt.plot(sorted_gen, label=l, color=c, lw=3)
plt.ylabel('Generation [MWh/h]')
plt.xlabel('Hours')
plt.title('Duration curve of generation')
plt.legend(fancybox=True, shadow=True, loc='best')
plt.tight_layout()
plt.savefig('1a_duration_curve.png', dpi=300)
plt.show()

# Normalised duration curves (capacity factors)
plt.figure(figsize=(8, 5), dpi=300)
for n, l, c in zip(model_names_a, labels_a, colors_a):
    sorted_cf = (network.generators_t.p[n].sort_values(ascending=False).reset_index(drop=True)
                 / network.generators.p_nom_opt[n])
    plt.plot(sorted_cf, label=l, color=c, lw=3)
plt.ylabel('Capacity factor [-]')
plt.xlabel('Hours')
plt.title('Normalised duration curve (capacity factors)')
plt.legend(fancybox=True, shadow=True, loc='best')
plt.tight_layout()
plt.savefig('1a_duration_curve_CFs.png', dpi=300)
plt.show()
