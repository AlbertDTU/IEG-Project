import pandas as pd
import numpy as np
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

# ── Build the base network (no CO2 constraint yet) ────────────────────────────
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

    network.add("StorageUnit", "battery storage",
                bus="electricity bus", carrier="battery storage",
                max_hours=2,
                capital_cost=annuity(20, 0.07) * 2 * 288000,
                efficiency_store=0.98, efficiency_dispatch=0.97,
                p_nom_extendable=True, cyclic_state_of_charge=True)

    return network

# ── CO2 sweep ─────────────────────────────────────────────────────────────────
# Range: from ~15 Mt (effectively unconstrained for Denmark) down to 0.5 Mt
co2_limits = np.concatenate([
    np.linspace(15e6, 1e6, 15),
    np.array([0.5e6])
])

gen_cols   = ["Onshore wind", "Offshore wind", "Solar", "OCGT", "CCGT", "battery storage"]
cap_cols   = ["Onshore wind", "Offshore wind", "Solar", "OCGT", "CCGT", "battery storage"]

capacity_results   = {}
generation_results = {}
cost_results       = {}
emission_results   = {}

network = build_network()

for co2_val in co2_limits:
    # Update the CO2 constraint
    if "co2_limit" in network.global_constraints.index:
        network.remove("GlobalConstraint", "co2_limit")
    network.add("GlobalConstraint", "co2_limit",
                type="primary_energy",
                carrier_attribute="co2_emissions",
                sense="<=",
                constant=co2_val)

    try:
        status, condition = network.optimize(solver_name='gurobi', solver_options={"OutputFlag": 0, "LogToConsole": 0})
    except Exception:
        print(f"  Infeasible at {co2_val/1e6:.2f} Mt — skipping")
        continue

    if status != "ok":
        print(f"  Solver status '{status}' at {co2_val/1e6:.2f} Mt — skipping")
        continue

    # Capacity [GW]
    cap = network.generators.p_nom_opt.copy()
    cap["battery storage"] = network.storage_units.p_nom_opt["battery storage"]
    capacity_results[co2_val] = cap[cap_cols].div(1e3)

    # Generation [TWh/yr]
    gen = network.generators_t.p.sum().copy()
    gen["battery storage"] = network.storage_units_t.p.clip(lower=0).sum()["battery storage"]
    generation_results[co2_val] = gen[gen_cols].div(1e6)

    # Total system cost [M€/yr]
    cost_results[co2_val] = network.objective / 1e6

    # CO2 emissions per gas generator [Mt CO2/yr]
    # primary_energy = generation / efficiency * co2_emissions_factor (0.19 t/MWh_th)
    emission_results[co2_val] = pd.Series({
        "OCGT": network.generators_t.p["OCGT"].sum() / 0.41 * 0.19 / 1e6,
        "CCGT": network.generators_t.p["CCGT"].sum() / 0.56 * 0.19 / 1e6,
    })

    print(f"  CO2 = {co2_val/1e6:.2f} Mt  |  cost = {cost_results[co2_val]:.1f} M€")

# ── Assemble DataFrames ────────────────────────────────────────────────────────
df_capacity   = pd.DataFrame(capacity_results).T    # index: tonnes, cols: GW
df_generation = pd.DataFrame(generation_results).T  # index: tonnes, cols: TWh
df_cost       = pd.Series(cost_results)              # index: tonnes, values: M€
df_emissions  = pd.DataFrame(emission_results).T     # index: tonnes, cols: Mt CO2

# Convert index to Mt CO2 for readable axis labels
df_capacity.index   = df_capacity.index / 1e6
df_generation.index = df_generation.index / 1e6
df_cost.index       = df_cost.index / 1e6
df_emissions.index  = df_emissions.index / 1e6

labels = ['Onshore wind', 'Offshore wind', 'Solar PV', 'Gas (OCGT)', 'Gas (CCGT)', 'Battery storage']
colors = ['blue', 'dodgerblue', 'orange', 'crimson', 'darkviolet', 'lightgreen']

CO2_REF = 8.27  # Mt — Denmark EU ETS 2024 verified emissions

def bar_vline(ax, index_vals, ref_val):
    """Add a vertical line at ref_val on a bar chart whose x-ticks are integers."""
    # Bar chart x-positions are 0,1,2,... corresponding to index_vals (high→low)
    # Interpolate to find the fractional position of ref_val
    positions = np.arange(len(index_vals))
    x_ref = np.interp(ref_val, index_vals[::-1], positions[::-1])
    ax.axvline(x=x_ref, color='black', linestyle='--', linewidth=1.5,
               label=f'{ref_val} Mt limit', zorder=5)

# ── Plot 1: Generation mix vs CO2 limit (bar) ────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
df_generation.plot.bar(stacked=True, ax=ax, color=colors, edgecolor='none', width=0.8)
bar_vline(ax, df_generation.index.values, CO2_REF)
ax.set_xlabel('CO₂ limit [Mt CO₂/yr]', fontsize=11)
ax.set_ylabel('Annual generation [TWh/yr]', fontsize=11)
ax.set_title('Generation mix vs. CO₂ constraint', fontweight='bold')
ax.set_xticklabels([f'{v:.1f}' for v in df_generation.index], rotation=45, ha='right')
ax.legend(labels + [f'{CO2_REF} Mt limit'], bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)
plt.tight_layout()
plt.savefig('part_f_generation_mix.png', dpi=300)
plt.show()

# ── Plot 1b: Generation mix vs CO2 limit (line) ──────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
for col, label, color in zip(df_generation.columns, labels, colors):
    ax.plot(df_generation.index, df_generation[col], marker='o', markersize=4,
            label=label, color=color, lw=2)
ax.axvline(x=CO2_REF, color='black', linestyle='--', linewidth=1.5, label=f'{CO2_REF} Mt limit')
ax.invert_xaxis()
ax.set_xlabel('CO₂ limit [Mt CO₂/yr]', fontsize=11)
ax.set_ylabel('Annual generation [TWh/yr]', fontsize=11)
ax.set_title('Generation mix vs. CO₂ constraint', fontweight='bold')
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('part_f_generation_mix_lines.png', dpi=300)
plt.show()

# ── Plot 2: Capacity mix vs CO2 limit (bar) ───────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
df_capacity.plot.bar(stacked=True, ax=ax, color=colors, edgecolor='none', width=0.8)
bar_vline(ax, df_capacity.index.values, CO2_REF)
ax.set_xlabel('CO₂ limit [Mt CO₂/yr]', fontsize=11)
ax.set_ylabel('Installed capacity [GW]', fontsize=11)
ax.set_title('Capacity mix vs. CO₂ constraint', fontweight='bold')
ax.set_xticklabels([f'{v:.1f}' for v in df_capacity.index], rotation=45, ha='right')
ax.legend(labels + [f'{CO2_REF} Mt limit'], bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)
plt.tight_layout()
plt.savefig('part_f_capacity_mix.png', dpi=300)
plt.show()

# ── Plot 2b: Capacity mix vs CO2 limit (line) ────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
for col, label, color in zip(df_capacity.columns, labels, colors):
    ax.plot(df_capacity.index, df_capacity[col], marker='o', markersize=4,
            label=label, color=color, lw=2)
ax.axvline(x=CO2_REF, color='black', linestyle='--', linewidth=1.5, label=f'{CO2_REF} Mt limit')
ax.invert_xaxis()
ax.set_xlabel('CO₂ limit [Mt CO₂/yr]', fontsize=11)
ax.set_ylabel('Installed capacity [GW]', fontsize=11)
ax.set_title('Capacity mix vs. CO₂ constraint', fontweight='bold')
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('part_f_capacity_mix_lines.png', dpi=300)
plt.show()

# ── Plot 3: System cost vs CO2 limit ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
ax.plot(df_cost.index, df_cost.values, marker='o', color='steelblue', lw=2)
ax.set_xlabel('CO₂ limit [Mt CO₂/yr]', fontsize=11)
ax.set_ylabel('Total annual system cost [M€/yr]', fontsize=11)
ax.set_title('System cost vs. CO₂ constraint', fontweight='bold')
ax.invert_xaxis()  # tighter constraint on the right
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('part_f_system_cost.png', dpi=300)
plt.show()

# ── Plot 4: CO2 emissions per generator vs CO2 limit ─────────────────────────
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
df_emissions.plot.bar(stacked=True, ax=ax, color=['crimson', 'darkviolet'],
                      edgecolor='none', width=0.8)

# Overlay the CO2 limit as a reference line (shows when constraint is binding)
x_positions = range(len(df_emissions))
ax.plot(x_positions, df_emissions.index, color='black', linestyle='--',
        linewidth=1.5, label='CO₂ limit', zorder=5)

ax.set_xlabel('CO₂ limit [Mt CO₂/yr]', fontsize=11)
ax.set_ylabel('CO₂ emissions [Mt CO₂/yr]', fontsize=11)
ax.set_title('CO₂ emissions per generator vs. CO₂ constraint', fontweight='bold')
ax.set_xticklabels([f'{v:.1f}' for v in df_emissions.index], rotation=45, ha='right')
ax.legend(['Gas (OCGT)', 'Gas (CCGT)', 'CO₂ limit'], bbox_to_anchor=(1.01, 1),
          loc='upper left', frameon=False)
plt.tight_layout()
plt.savefig('part_f_co2_emissions.png', dpi=300)
plt.show()
