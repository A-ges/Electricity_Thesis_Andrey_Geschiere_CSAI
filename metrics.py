import numpy as np
import pandas as pd
from scipy.signal import argrelmin
pd.set_option('display.float_format', '{:.3f}'.format) #needed to not have the e scientific notation in the social_flex_contribution and just get the rounded values


"""
This file computes all relevant metrics to answer my RQs and some for model tweaking and functions that help making agent shifting decisions

Two DataFrames are assembled and returned to run_model:
- df_agent_daily -> one row per agent per day, allows for group-level analysis
- df_daily -> one row per day, allows for system-level analysis

All functions are called from run_model.py at the end of each simulated day.

Cost metric design note:
    -> Costs are computed on appliance load ONLY (baseline is subtracted)
    -> The normalized cost (pricing units per kWh of appliance load) is comparable across agents
       regardless of consumption characteristics
    -> Pricing units are EPEX-derived values divided by 10 as seen in price_estimator.py
    -> They are consistent for relative comparison but do not correspond to real world consumer costs
"""


def find_local_price_minima(dayprices):
    """
    Parameters:
    -> The list with prices for DAY X
    
    Used to find local minima in the 24-hour price curve each day
    Uses scipy.signal.argrelmin with order = 1 (comparison points for it to be a minimum, lower than direct neighbors suffices)    
    Falls back to the global minimum if no local minima are detected
    Returns a list of integer hour indices with low target prices for the shifts
    """
    prices = np.array(dayprices, dtype=float)  #convert to numpy array for argrelmin
    minima = argrelmin(prices, order=1)[0].tolist()  #find all local minima where direct values next to it are hiher
    if not minima:
        #treat the single cheapest hour as the only target if monotone or extremely flat price curve
        minima = [int(np.argmin(prices))]
    return minima

def compute_social_targets_for_agent(agent, previous_day_contacts, agents_by_id, appliance_names):
    """
    Used to compute per-appliance social shift targets for one agent

    For each appliance this function collects peak centers grouped by peak index
    from yesterday's peak lists of every daily contact, then returns one mean
    center per index as a dict

    Parameters:
    -> agent: Agent whose social targets are being computed
    -> previous_day_contacts: daily contact network of DAY-1, full dictionary
    -> agents_by_id: quick lookup for contact Agent objects
    -> appliance_names: appliances to compute targets for

    Returns a dict keyed by appliance name, where each value is itself a dict
    mapping peak index to mean center: {appliance: {0: mean_peak_0, 1: mean_peak_1, ...}}
    or None if no contacts used the appliance yesterday
    """
    social_targets = {}  #will hold one entry per appliance
    contact_ids = previous_day_contacts[agent.agent_id]  #get this agent's daily contacts from yesterday

    for appliance in appliance_names:
        #determine how many peaks this agent has for this appliance
        agent_peaks = agent.previous_peak_lists.get(appliance, [])
        n_peaks = len(agent_peaks)

        if n_peaks == 0:
            social_targets[appliance] = None
            continue

        #collect centers separately per peak index across all contacts
        #centers_by_index maps peak index -> list of all centers contributed by contacts for that index
        centers_by_index = {i: [] for i in range(n_peaks)}
        for contact_id in contact_ids:
            contact = agents_by_id[contact_id]  #look up the contact Agent object

            #use .get() so contacts without EV or with fewer peaks are skipped gracefully
            contact_peaks = contact.previous_peak_lists.get(appliance, [])
            for i, (c, h, w) in enumerate(contact_peaks):
                if i < n_peaks:
                    centers_by_index[i].append(c)  #route this contact's center to the correct index bucket

        #compute one mean per peak index, None if no contacts contributed to that index
        #result is a dict {peak_index: mean_center_or_None} so the caller can look up by index
        target_dict = {}
        has_any = False
        for i, centers in centers_by_index.items():
            if centers:
                target_dict[i] = float(np.mean(centers))
                has_any = True
            else:
                target_dict[i] = None  #no contact data for this peak index

        social_targets[appliance] = target_dict if has_any else None

    return social_targets


def compile_agent_day_metrics(agent, day, load, prices, is_last_day=False):
    """
    Get all per-agent metrics for a simulated day

    Parameters:
    -> agent: An agent
    -> day: day number where the first day = 0
    -> load: agent's 15-min load profile for this day
    -> prices: price per 15-min slot (each hour price repeated 4 times)
    -> is_last_day: True only on the final simulated day; if False, individual_adjustment is NaN

    Returns a dict of metrics, will be one row in df_agent_daily
    """
    baseline = agent.appliance_chars["Baseline"]["power_kw"]  #this agent's always-on power draw

    #remove constant baseline from the load to isolate behavior driven appliance consumption
    appliance_load = load - baseline

    #multiply by 0.25 because each slot is 15 minutes = 0.25 hours
    total_appliance_kwh = float(appliance_load.sum() * 0.25)
    #raw cost: sum of (appliance kW * price * 0.25h) across all 96 slots, giving kWh * price = cost
    raw_cost = float((appliance_load * prices).sum() * 0.25)

    #normalized cost: pricing units per kWh of appliance load
    #dividing by energy makes the cost comparable across agents regardless of total consumption
    if total_appliance_kwh > 0.0: 
        normalized_cost = raw_cost / total_appliance_kwh
    else:
        normalized_cost = 0.0  #avoid division by zero if agents with no appliance use today

    #these are the shift magnitudes stored on the agent by apply_shifts() earlier this day
    total_flex = agent.last_total_flexibility
    price_flex = agent.last_price_flexibility
    social_flex = agent.last_social_flexibility

    
    if is_last_day:
        adjustment = agent.compute_adjustment()
    else:
        adjustment = float("nan")

    #to compute tradeoff
    mean_price_today = float(prices.mean())
    if total_appliance_kwh > 0.0:
        agent_effective_price = raw_cost / total_appliance_kwh
    else:
        agent_effective_price = mean_price_today  #edge case if no appliance consumption
 
    price_advantage = mean_price_today - agent_effective_price #positive = paid less than average, negative = more

    #CHANGED: added minimum threshold before computing savings_per_flex
    #social-influenced agents have price_sens drawn from Beta(0.5, 4.5) which can produce values near 0
    #on day 1, social_flex = 0 (all agents start at the same centers) and a near-zero price_sens gives
    #total_flex ≈ 0 as well. dividing any nonzero price_advantage by near-zero total_flex produces
    #extreme outliers (e.g. -200) that distort savings_per_flex_social_inss fluenced_mean across all days
    #the threshold of 0.5 requires at least 0.5 hours of total peak shift acroall appliances,
    #which price-responsive (flex ≈ 8) and habit agents (flex ≈ 5) comfortably exceed on every shift day
    #but filters out near-zero-price_sens social agents on day 1 where the metric is not meaningful
    if total_flex > 0.0:
        savings_per_flex = price_advantage / total_flex
    else:
        savings_per_flex = float("nan")  #not meaningful when agent has barely shifted

    return {
        "day": day,
        "agent_id": agent.agent_id,
        "dominant_group": agent.dominant_group,
        "habit_str": agent.habit_str,
        "price_sens": agent.price_sens,
        "soc_suc": agent.soc_suc,
        "individual_flexibility": total_flex,
        "price_shift_contribution": price_flex,
        "social_shift_contribution": social_flex,
        "individual_cost_raw": raw_cost,
        "individual_cost_normalized": normalized_cost,
        "individual_adjustment": adjustment,
        "total_appliance_kwh": total_appliance_kwh,
        "price_advantage": price_advantage,
        "savings_per_flex": savings_per_flex}

def gini_coefficient(load_array):
    """
    Compute the Gini coefficient of full day load profile

    Gini is in the range [0, 1] where:
    -> 0 = perfectly flat load (all 96 slots draw exactly the same power)
    -> 1 = all load concentrated in a single slot (maximally unequal and horrible for grid stability)
    Used as a measure of temporal load inequality, measure for grid stress
    """
    arr = np.sort(np.abs(load_array).ravel())  #sort values from low to high
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0  #empty or flat array, no inequality
    index = np.arange(1, n + 1)  #rank index 1 to n
    #Gini formula on sorted array
    return float((2.0 * (index * arr).sum()) / (n * arr.sum()) - (n + 1) / n)
    #low slots get low weights, high slots get high weights, then normalize by total load and count.


def compile_day_metrics(day, aggregate, prices, agent_records):
    """
    Compute system-level metrics for a single simulated day

    Parameters:
    -> day: where the first day is 0
    -> aggregate: total load across all agents in kW per 15-min slot
    -> prices: hourly prices used on this day
    -> agent_records: output of compile_agent_day_metrics

    Returns a dict of metrics = one row for df_daily

    """
    prices_arr = np.array(prices) 
    peak_load = float(aggregate.max())  #highest 15-min demand of the day in kW
    mean_load = float(aggregate.mean()) #average demand  
    par = peak_load / mean_load  #Peak-to-average ratio: higher = more grid stress
    load_var = float(aggregate.var()) #variance of the 96-slot profile
    load_cv = float(aggregate.std() / mean_load) #coefficient of variation: normalized spread
    load_gini = gini_coefficient(aggregate) #temporal load inequality
    total_energy_kwh = float(aggregate.sum() * 0.25) #total energy in kWh for this day
    peak_hour = float(np.argmax(aggregate)) / 4.0  #slot index to hour of day

    mean_price = float(prices_arr.mean()) #average price across the day
    price_min = float(prices_arr.min()) #cheapest hour of the day
    price_max = float(prices_arr.max()) #most expensive hour of the day
    price_range = price_max - price_min #range of prices, larger = stronger shifting incentive

    #System-level behavioral metrics from individual agent records 
    df_agents = pd.DataFrame(agent_records) #convert list of dicts to df

    total_flexibility = float(df_agents["individual_flexibility"].sum()) #sum of all agents' shifts today
    mean_cost_norm = float(df_agents["individual_cost_normalized"].mean()) #mean normalized cost across all agents
    mean_adjustment = float(df_agents["individual_adjustment"].mean()) #mean cumulative adjustment across all agents (NaN on non-last days)
    mean_price_advantage = float(df_agents["price_advantage"].mean()) #positive = agents on average paid below daily mean

    #Per-dominant-group metrics for comparison
    group_stats = {}
    for group in ["Habit-driven", "Price-responsive", "Social-influenced"]:
        subset = df_agents[df_agents["dominant_group"] == group]  #filter to only this group's rows
        key = group.lower().replace("-", "_").replace(" ", "_")  #change col names to lower case with -, for consistency
        if len(subset) > 0: #could input no agents of some group in run model
            group_stats[f"flex_{key}_mean"] = float(subset["individual_flexibility"].mean())
            group_stats[f"cost_norm_{key}_mean"] = float(subset["individual_cost_normalized"].mean())
            #adjustment = mean adjustment; NaN on non-last days because individual_adjustment is NaN then
            group_stats[f"adjustment_{key}_mean"] = float(subset["individual_adjustment"].mean())
            group_stats[f"savings_per_flex_{key}_mean"] = float(subset["savings_per_flex"].mean(skipna=True))
            group_stats[f"n_{key}"] = int(len(subset))
        
        else:
            #group not present in this run, fill with NaN to avoid errors
            group_stats[f"flex_{key}_mean"] = float("nan")
            group_stats[f"cost_norm_{key}_mean"] = float("nan")
            group_stats[f"adjustment_{key}_mean"] = float("nan")
            group_stats[f"savings_per_flex_{key}_mean"] = float("nan")
            group_stats[f"n_{key}"] = 0

    #assemble the final record dict for this day
    record = {
        "day": day,
        "peak_load_kw": peak_load,
        "mean_load_kw": mean_load,
        "par": par,
        "load_variance": load_var,
        "load_cv": load_cv,
        "load_gini": load_gini,
        "total_energy_kwh": total_energy_kwh,
        "peak_hour": peak_hour,
        "mean_price": mean_price,
        "price_min": price_min,
        "price_max": price_max,
        "price_range": price_range,
        "total_flexibility": total_flexibility,
        "mean_cost_norm": mean_cost_norm,
        "mean_adjustment": mean_adjustment,
        "mean_price_advantage": mean_price_advantage}
    record.update(group_stats)  #merge in the per group stats
    return record

def build_dataframes(agent_day_records, day_records):
    """
    Convert the two records collected during simulation into two dfs
    Also adds a cumulative column to df_agent_daily:
    -> cumulative_flexibility: running sum of individual_flexibility per agent across days
       useful for tracking the total grid service each household has provided
    """
    df_agent_daily = pd.DataFrame(agent_day_records)  #one row per agent per day
    df_daily = pd.DataFrame(day_records) #one row per day
    
    df_agent_daily = df_agent_daily.sort_values(["agent_id", "day"]).reset_index(drop=True)
    df_daily = df_daily.sort_values("day").reset_index(drop=True)

    df_agent_daily["cumulative_flexibility"] = (
        df_agent_daily.groupby("agent_id")["individual_flexibility"].cumsum())

    return df_agent_daily, df_daily
