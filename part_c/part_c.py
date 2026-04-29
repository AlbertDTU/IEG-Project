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

# ── Build network (generators only, no storage) ────────────────────────────────
def build_network():
    network = pypsa.Network()
    hours_in_2015 = pd.date_range('2015-01-01 00:00Z', '2015-12-31 23:00Z', freq='h')
    network.set_snapshots(hours_in_2015.values)
    network.add("Bus", "electricity bus")
    network.add("Load", "load", bus="electricity bus", p_set=df_elec[country].values)

    network.add("Carrier", "gas", co2_emissions=0.19)
    network.add("Carrier", "onshorewind")
    network.add("Carrier", "offshorewind")
    network.add("Carrier", "solar")
    network.add("Carrier", "battery storage")

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

    return network


labels      = ['Onshore wind', 'Offshore wind', 'Solar PV', 'Gas (OCGT)', 'Gas (CCGT)', 'Battery storage']
model_names = ['Onshore wind', 'Offshore wind', 'Solar',    'OCGT',       'CCGT',       'battery storage']
colors      = ['blue', 'dodgerblue', 'orange', 'crimson', 'darkviolet', 'lightgreen']

# ── Scenario A: no storage (baseline for comparison) ──────────────────────────
network_no_storage = build_network()
network_no_storage.optimize(solver_name='gurobi', solver_options={"OutputFlag": 0, "LogToConsole": 0})

print("=== Without storage ===")
print(f"Total system cost: {network_no_storage.objective / 1e6:.2f} M€/yr")
print("Optimal capacities [GW]:")
print(network_no_storage.generators.p_nom_opt.div(1e3))

# ── Scenario B: with battery storage ──────────────────────────────────────────
network = build_network()
network.add(
    "StorageUnit", "battery storage",
    bus="electricity bus", carrier="battery storage",
    max_hours=2,
    capital_cost=annuity(20, 0.07) * 2 * 288000,
    efficiency_store=0.98, efficiency_dispatch=0.97,
    p_nom_extendable=True, cyclic_state_of_charge=True,
)
network.optimize(solver_name='gurobi', solver_options={"OutputFlag": 0, "LogToConsole": 0})

# Merge generator and storage optimal capacities for reporting
optimal_cap = network.generators.p_nom_opt.copy()
optimal_cap['battery storage'] = network.storage_units.p_nom_opt['battery storage']

print("\n=== With battery storage ===")
print(f"Total system cost: {network.objective / 1e6:.2f} M€/yr")
print(f"Cost reduction vs no storage: {(network_no_storage.objective - network.objective) / 1e6:.2f} M€/yr")
print("\nOptimal capacities [GW]:")
print(optimal_cap.div(1e3))
print(f"\nBattery capacity: {network.storage_units.p_nom_opt['battery storage'] / 1e3:.2f} GW  "
      f"({network.storage_units.p_nom_opt['battery storage'] * 2 / 1e3:.2f} GWh)")

# ── Plots ──────────────────────────────────────────────────────────────────────

# Installed capacity pie chart (with storage)
cap_sizes = [optimal_cap[n] for n in model_names]
plt.figure(figsize=(6, 5), dpi=300)
plt.pie(cap_sizes, colors=colors,
        labels=[f'{l}\n{s/1e3:.1f} GW' for l, s in zip(labels, cap_sizes)],
        wedgeprops={'linewidth': 0})
plt.axis('equal')
plt.title('Installed capacity mix (with battery storage)', y=1.05, fontweight='bold')
plt.tight_layout()
plt.savefig('1c_installed_capacity_mix.png', dpi=300)
plt.show()

# Battery storage stackplot — full year
fig, ax = plt.subplots(figsize=(12, 4), dpi=300)

bat_dis = network.storage_units_t.p.clip(lower=0)   # discharge → positive
bat_ch  = network.storage_units_t.p.clip(upper=0)   # charge    → negative

renewables = network.generators_t.p[['Onshore wind', 'Offshore wind', 'Solar']].sum(axis=1)
supply = pd.concat([
    bat_dis.rename(columns={"battery storage": "Battery discharge"}),
    renewables.to_frame("Renewables"),
], axis=1)
supply.plot.area(ax=ax, linewidth=0, stacked=True, color=["#2ca02c", "#1f77b4"])
bat_ch.rename(columns={"battery storage": "Battery charge"}).plot.area(
    ax=ax, linewidth=0, stacked=True, color=["#ff7f0e"])
network.loads_t.p_set.sum(axis=1).plot(ax=ax, color="black", linestyle="--", label="Demand")

ax.axhline(0, color="black", linewidth=0.5)
ax.set_ylabel("MW")
ax.set_title("Battery storage behaviour — full year")
ax.legend(frameon=False, bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('1c_battery_stackplot_year.png', dpi=300)
plt.show()

# Battery state of charge — one summer week and one winter week
fig, axes = plt.subplots(2, 1, figsize=(10, 6), dpi=300)
soc = network.storage_units_t.state_of_charge['battery storage']

for ax, (start, end, season) in zip(axes, [(0, 168, 'Winter (Jan)'), (4344, 4512, 'Summer (Jul)')]):
    soc.iloc[start:end].plot(ax=ax, color='lightgreen', lw=2)
    ax.set_ylabel('State of charge [MWh]')
    ax.set_title(f'Battery state of charge — {season}')
    ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('1c_battery_soc.png', dpi=300)
plt.show()

# Dispatch stackplot — one summer week and one winter week
gen_names = ['Onshore wind', 'Offshore wind', 'Solar', 'OCGT', 'CCGT']
gen_colors = ['blue', 'dodgerblue', 'orange', 'crimson', 'darkviolet']

for start, end, season, fname in [
    (0,    168,  'Winter (Jan)', '1c_dispatch_winter.png'),
    (4344, 4512, 'Summer (Jul)', '1c_dispatch_summer.png'),
]:
    fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
    active_gens = [n for n in gen_names if network.generators.p_nom_opt[n] > 0]
    active_colors = [c for n, c in zip(gen_names, gen_colors) if network.generators.p_nom_opt[n] > 0]
    stack_data = [network.generators_t.p[n].iloc[start:end].values for n in active_gens]
    ax.stackplot(network.snapshots[start:end], *stack_data,
                 labels=active_gens, colors=active_colors, linewidth=0)
    bat_dis_week = bat_dis['battery storage'].iloc[start:end]
    bat_ch_week  = bat_ch['battery storage'].iloc[start:end]
    ax.fill_between(network.snapshots[start:end], bat_dis_week.values,
                    label='Battery discharge', color='#2ca02c', alpha=0.8)
    ax.fill_between(network.snapshots[start:end], bat_ch_week.values,
                    label='Battery charge', color='#ff7f0e', alpha=0.8)
    network.loads_t.p['load'].iloc[start:end].plot(ax=ax, color='black', lw=2, label='Demand')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel('MW')
    ax.set_title(f'Dispatch with battery storage — {season}')
    ax.legend(frameon=False, bbox_to_anchor=(1.05, 1), fontsize=8)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.show()
