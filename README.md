# From Household Decisions to Electric Grid Patterns

> Bachelor Thesis | Cognitive Science & Artificial Intelligence | Tilburg University.

## Overview

An agent-based model (ABM) of residential electricity demand under a variable-priced
Western-European grid. The model simulates a population of heterogeneous households
over a self defined period at 15-minute granularity. Each household is built up from
appliance-level usage and is assigned one of three behavioral architectures:

- **Habit-driven** -> concentrate consumption on preferred hours, low responsiveness
- **Price-responsive** -> shift load towards local price minima
- **Social-influenced** -> gradually adopt the schedules observed within their daily contact network

Households interact through a pre-built social network and respond
to a dynamic price signal derived from EPEX day-ahead baselines and a
solar-elasticity factor calibrated on the OpenSTEF Liander 2024 dataset.

## Repository Structure

```text
📁 Project Root
├── 📄 run_model.py              # main simulation entry point
├── 📄 agent.py                  # Agent class + shifting logic
├── 📄 Setting_Parameters.py     # behavioral parameter sampling (Beta distributions)
├── 📄 load_profile.py           # appliance distributions and daily load generation
├── 📄 generate_daily_contacts.py# daily contact sub-network sampling
├── 📄 price_estimator.py        # hourly price model (demand + solar elasticity)
├── 📄 metrics.py                # agent- and system-level metric computation
├── 📄 runhere.ipynb             # quick-start notebook with a small example run
├── 🗜️ Network_Zipped.zip        # pregenerated networks (see Setup)
├── 📄 requirements.txt
├── 📄 README.md
│
├── 📁 groundwork                # precompute scripts run once before simulation
│   ├── 📄 make_network.py       # Hamiltonian network builder
│   ├── 📄 Baseline_Distributions.py  # Gaussian peak fitting on Yilmaz et al. (2017)
│   ├── 📄 run_baseline.py       # no-shift baseline reference runs
│   └── 📁 price_model_baselines # EPEX baseline and solar elasticity computation
│
├── 📁 analysis                  # notebooks producing all thesis figures and stats
│   ├── 📄 qualitativeanalysis.ipynb  # qualitative analysis (Section 5.1)
│   ├── 📄 RQ1.ipynb             # within-population group differences (RQ1)
│   ├── 📄 RQ2.ipynb             # system-level composition effects (RQ2)
│   ├── 📁 datagenerators        # scripts that produce the cached results
│   │   ├── 📄 datagenerator.ipynb     # main grid runner
│   │   ├── 📄 RQ1figgen.ipynb         # cost-comfort figure data
│   │   └── 📄 RQ2fig1gen.ipynb        # day-29 aggregate curves
│   ├── 📁 results               # cached outputs from the datagenerators
│
└── 📁 project_report            # LaTeX backup of the thesis
```

> The notebooks in `analysis/` are working notebooks they generate the figures
> and statistics used in the thesis. But note that the intermediate
> reasoning, and personal notes are NOT final and outdated.
> Everything regarding true and relevant analysis can be found in the thesis itself  

## Setup

1. **Python version.** Developed on Python 3.12.8.

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Unzip the pre-built networks.**
   `networks.json` was too large to commit directly, so it's zipped as
   `Network_Zipped.zip`. Unzip it and place `networks.json` in the project root:
   ```bash
   unzip Network_Zipped.zip
   ```

4. **(Optional) Regenerate `networks.json` yourself.**
   `groundwork/make_network.py` rebuilds the full set of networks for
   N ∈ {50, 100, ..., 1000} with five variants each {a, b, c, d, e}. This takes a long time
   (up to a day on a single machine) and is not required if you use the
   unzipped file from step 3.

5. **Paths.** The code assumes the project root as the working directory.
   Datagenerator notebooks write to `analysis/results/` and analysis notebooks
   read from there.

## Running the Model

I added the runhere.ipynb file for easy running of a simulation.
An example run:

```python
from run_model import run_model

df_agents, df_daily, load_profiles, df_pricing = run_model(
    agents_pct   = [60, 30, 10],   # [habit%, price%, social%], must sum to 100
    network_code = "500a",         # one of {50a, 50b, …, 1000e}
    days         = 30,
    random_state = 42)
```

This returns four pandas DataFrames / numpy arrays:

- `df_agents` — per agent, per day metrics (flexibility, cost, adjustment, ...)
- `df_daily` — per-day system metrics (PAR, mean price, ...)
- `load_profiles` — `(days, 96)` aggregate kW per 15-minute slot
- `df_pricing` — per-day, per-hour price (baseline vs used)
  
> Some metrics (e.g. Gini, variance), were implemented but not used in the final thesis

`runhere.ipynb` shows the same example with a small population (`300c`, 10 days)
and prints output shapes. Refer to the docstring of `run_model` for all
configurable arguments, including the three epsilon parameters that control
the strength of the habit, price, and social shifting channels.

## Environment

| Component | Version |
| --- | --- |
| Python | 3.12.8 |
| NumPy | 1.26.4 |
| Pandas | 2.2.2 |
| SciPy | 1.13.1 |
| Matplotlib | 3.9.2 |
| Statsmodels | 0.14.2 |
| huggingface_hub | only required to get Liander parquet files in the groundwork folder |

## Data Sources

All synthetic data analysed in the thesis was generated with the model in this repository.

Empirical baselines used in the model:

- **OpenSTEF Liander 2024** -> 15-minute residential demand, EPEX day-ahead prices,
  and solar park generation for five Dutch stations.
- **Yilmaz et al. (2017)** -> per-appliance hourly switch-on probabilities.
- **Robinson et al. (2013)** -> EV charging power, duration, and switch-on curve.
- **Williams et al. (2025)** -> appliance power and runtime distributions, baseline
  load magnitude, and the bottom-up methodological framework.

For full references and an understanding of the methodology, refer to the thesis.
