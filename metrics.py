import numpy as np
import pandas as pd
from scipy.signal import find_peaks

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
    -> Pricing units are EPEX-derived values divided by 10 as set in price_estimator.py
    -> They are consistent for relative comparison but do not correspond to real world consumer costs
"""


def find_local_price_minima(dayprices):
    """
    Parameters:
    -> The list with prices for DAY X
    
    Used to find local minima in the 24-hour price curve each day
    Uses scipy.signal.argrelmin with order = 1 (comparison points for it to be a minimum, lower than direct neighbors suffices)
    For documentation refer to: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.argrelmin.html
    
    Falls back to the global minimum if no local minima are detected
    Returns a list of integer hour indices with low target prices for the shifts
    """
    from scipy.signal import argrelmin
    prices = np.array(dayprices, dtype=float)  #convert to numpy array for argrelmin
    minima = argrelmin(prices, order=1)[0].tolist()  #find all strict local minima
    if not minima:
        #rare fallback: treat the single cheapest hour as the only target if monotone or extremely flat price curve
        minima = [int(np.argmin(prices))]
    return minima

def compute_social_targets_for_agent(agent, previous_day_contacts, agents_by_id, appliance_names):
    """
    Used to compute per-appliance social shift targets for one agent

    For each appliance this function collects all peak hours from yesterday's
    peak lists of every daily contact and then returns the mean center

    Parameters:
    -> agent: Agent whose social targets are being computed
    -> previous_day_contacts: daily contact network of DAY-1, full dictionary
    -> agents_by_id: quick lookup for contact Agent objects
    -> appliance_names: appliances to compute targets for

    Returns a dict with mean peak center of all contacts for that appliance
    """
    social_targets = {}  #will hold one entry per appliance
    contact_ids = previous_day_contacts[agent.agent_id]  #get this agent's daily contacts from yesterday

    for appliance in appliance_names:
        centers = []  #collect all peak centers from all contacts for this appliance
        for contact_id in contact_ids:
            contact = agents_by_id[contact_id]  #look up the contact Agent object
            
            #use previous peaklists, based on the day of interaction
            #use .get() so contacts without EV are skipped
            for c, h, w in contact.previous_peak_lists.get(appliance, []):
                centers.append(c)  #add the center of this peak

        if centers:
            social_targets[appliance] = float(np.average(centers)) #average the centers
        else:
            social_targets[appliance] = None  #no contacts used this appliance (e.g. EV for non-EV contacts)

    return social_targets


def compile_agent_day_metrics(agent, day, load, prices):
    """
    Get all per-agent metrics for a simulated day

    Parameters:
    -> An agent
    -> day: day number where the first day = 0
    -> load: agent's 15-min load profile for this day
    -> prices: price per 15-min slot (each hour price repeated 4 times)

    Returns a dict of metrics, will be one row in df_agent_daily
    """
    baseline = agent.appliance_chars["Baseline"]["power_kw"]  #this agent's always-on power draw

    #remove constant baseline from the load to isolate bahavior driven appliance consumption
    appliance_load = load - baseline

    #multiply by 0.25 because each slot is 15 minutes = 0.25 hours
    total_appliance_kwh = float(appliance_load.sum() * 0.25)
    #raw cost: sum of (appliance kW * price * 0.25h) across all 96 slots, giving kWh * price = cost
    raw_cost = float((appliance_load * prices).sum() * 0.25)

    #normalized cost: pricing units per kWh of appliance load
    #dividing by energy makes the cost comparable across agents regardless of total consumption (which could be different because of the sampling in agent.py
    if total_appliance_kwh > 0.0: 
        normalized_cost = raw_cost / total_appliance_kwh
    else:
        normalized_cost = 0.0  #avoid division by zero if agents with no appliance use today

    #these are the shift magnitudes stored on the agent by apply_shifts() earlier this day
    total_flex = agent.last_total_flexibility
    price_flex = agent.last_price_flexibility
    social_flex = agent.last_social_flexibility

    #cumulative drift of peak centers away from the agent's own day-0 preferred positions
    discomfort = agent.compute_discomfort()

    #Cost-flexibility ratio
    #higher ratio = more costly per unit of flexibility provided to the grid
    #undefined (NaN) on day 0 because flexibility is 0 -> ratio would be division by zero
    if total_flex > 0.0:
        cost_flex_ratio = normalized_cost / total_flex
    else:
        cost_flex_ratio = float("nan")

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
        "individual_discomfort": discomfort,
        "total_appliance_kwh": total_appliance_kwh,
        "cost_flex_ratio": cost_flex_ratio}

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

def count_rebound_peaks(aggregate, prominence):
    """
    Count the number of prominent local maxima in the aggregate 96 slot load profile

    Uses scipy.signal.find_peaks with a prominence filter so that minor bumps are excluded
    The prominence threshold is a fraction of the day's mean load,
    making it scale invariant with N

    Parameters:
    -> aggregate: system aggregate load array per 15-min slot
    -> prominence: minimum peak prominence as a fraction of mean load
        -> 0.5 means a peak must stand at least 50% of mean load above its surroundings

    Returns number of prominent peaks detected on this day
    """
    meanload = aggregate.mean()
    needed_prominence = prominence * meanload  #convert fractional threshold to absolute kW
    peaks, _ = find_peaks(aggregate, prominence=needed_prominence)  #detect prominent enough maxima
    return len(peaks)


def compile_day_metrics(day, aggregate, prices, agent_records, prominence):
    """
    Compute system-level metrics for a single simulated day

    Parameters:
    -> day: where the first day is 0
    -> aggregate: total load across all agents in kW per 15-min slot
    -> prices: hourly prices used on this day
    -> agent_records: output of compile_agent_day_metrics
    -> prominence: prominence threshold passed to count_rebound_peaks

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
    n_peaks = count_rebound_peaks(aggregate, prominence) #number of prominent peaks (rebound detection)
    peak_hour = float(np.argmax(aggregate)) / 4.0  #slot index to hour of day

    mean_price = float(prices_arr.mean()) #average price across the day
    price_min = float(prices_arr.min()) #cheapest hour of the day
    price_max = float(prices_arr.max()) #most expensive hour of the day
    price_range = price_max - price_min #range of prices, larger = stronger shifting incentive

    #System-level behavioral metrics from individual agent records 
    df_agents = pd.DataFrame(agent_records) #convert list of dicts to df
    accum_flexibility = float(df_agents["individual_flexibility"].sum()) #get agents shifts today
    accum_cost_norm = float(df_agents["individual_cost_normalized"].mean()) #mean normalized cost across all agents
    mean_discomfort = float(df_agents["individual_discomfort"].mean()) #mean cumulative discomfort across all agents

    #Per-dominant-group metrics for comparison
    group_stats = {}
    for group in ["Habit-driven", "Price-responsive", "Social-influenced"]:
        subset = df_agents[df_agents["dominant_group"] == group]  #filter to only this group's rows
        key = group.lower().replace("-", "_").replace(" ", "_")  #change col names to lower case with a -, for consistency
        if len(subset) > 0: #could input no agents of some group in run model
            group_stats[f"flex_{key}_mean"] = float(subset["individual_flexibility"].mean())
            group_stats[f"cost_norm_{key}_mean"] = float(subset["individual_cost_normalized"].mean())
            group_stats[f"discomfort_{key}_mean"] = float(subset["individual_discomfort"].mean())
            group_stats[f"cost_flex_ratio_{key}_mean"] = float(subset["cost_flex_ratio"].mean(skipna=True))  #skipna for day-0 NaN
            group_stats[f"n_{key}"] = int(len(subset))
        else:
            #group not present in this run, fill with NaN to avoid errors
            group_stats[f"flex_{key}_mean"] = float("nan")
            group_stats[f"cost_norm_{key}_mean"] = float("nan")
            group_stats[f"discomfort_{key}_mean"] = float("nan")
            group_stats[f"cost_flex_ratio_{key}_mean"] = float("nan")
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
        "n_rebound_peaks": n_peaks,
        "peak_hour": peak_hour,
        "mean_price": mean_price,
        "price_min": price_min,
        "price_max": price_max,
        "price_range": price_range,
        "accumulative_flexibility": accum_flexibility,
        "accumulative_cost_norm_mean": accum_cost_norm,
        "mean_discomfort": mean_discomfort}
    record.update(group_stats)  #merge in the per group stats
    return record

def build_dataframes(agent_day_records, day_records):
    """
    Convert the two records collected during simulation into two dfs
    Also adds a cumulative column to df_agent_daily:
    -> cumulative_flexibility: total of flexibility per agent across days
    """
    df_agent_daily = pd.DataFrame(agent_day_records)  #one row per agent per day)
    df_daily = pd.DataFrame(day_records) #one row per day
    
    df_agent_daily = df_agent_daily.sort_values(["day", "agent_id"]).reset_index(drop=True)
    df_daily = df_daily.sort_values("day").reset_index(drop=True)

    #add cumulative flexibility per agent over the simulation period
    #useful for tracking the total grid service each household has provided
    df_agent_daily["cumulative_flexibility"] = (
        df_agent_daily.sort_values("day").groupby("agent_id")["individual_flexibility"].cumsum().values)

    return df_agent_daily, df_daily
