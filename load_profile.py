import numpy as np
import matplotlib.pyplot as plt

"""
load_profile.py is the file that determines when, how often, how intense, how long appliances are ran
Present Functions:
- def multi_peak_distribution(peak_list, baseline_probability=0.005) -> Makes a peak list into a probability distribution
- def adjust_peaks(peak_list) -> Used to calibrate baseline peaks to Liander data
- def sample_agent_appliances(random_state) -> To get fixed parameters on runtime, intensity (power draw), if agents has EV, sets agent baseline
- def build_daily_load(agent_appliances, has_ev, random_state, previous_overflow=None, custom_distributions=None) -> set schedules for appliances

The original methodology is adopted from Williams et al. (2025). 
For their implementation, refer to:
https://github.com/alikazemian-bot/AMPED-Residential/blob/main/AMPED-Residential%20Agent-based%20Model%20for%20Predicting%20Electricity%20Demand.ipynb

Methodology
- Each appliance has a 24h probability distribution -> determines WHEN it is likely to be switched on
- The number of daily uses is determined by iterating over the switch-on data sourced from Yilmaz et al. (2017),
  which draws on the UK Household electricity survey. On every value, a random float is drawn between 0-1,
  if this number exceeds the value, the times this appliance will be used on this day will go +1
  The outcome is independent of previous trials.
- Once a start-hour is sampled from the distribution, a random quarter within that hour is added to get a 15-minute resolution start time
- Intensity and length are sampled once per agent at initialisation and stay fixed
  using the characteristics sourced from the code by Williams et al. (2025)
- A flat baseline load represents always-on devices (e.g. fridge, router, standby). This load is evenly spread
  over the day (doesn't include rhythmic cycling like in Williams et al. (2025) implementation because that is
  not very behavior influenced). Size of the baseline is sampled per agent and was set by combining Williams et al.
  (2025) and calibrated on Liander aggregate data of a residential area

"""


#----------------------------------------------------------------------------------
#Switch-on probability data per hour
#Same data as in the Baseline_Distributions file, sourced from Yilmaz et al. (2017)
#Each list has 24 values where index 0 = 00:00-01:00 and index 23 = 23:00-00:00
#Used only to determine how many times per day each appliance is activated
#The actual start-hour is drawn from the Gaussian distributions, not from this
#----------------------------------------------------------------------------------

data = {
    "Dishwasher":   [0.016,0.0071,0.0071,0.0044,0.0044,0.0017,0.008,0.0266,0.038,0.032,0.03,0.021,0.024,0.034,0.025,0.025,0.0196,0.027,0.0577,0.06577,0.0524,0.0328,0.0488,0.0355],
    "Washing":[0.00525,0.00617,0.00795,0.00527,0.00354,0.01777,0.03021,0.11463,0.15107,0.2061,0.2213,0.1812,0.1412,0.12976,0.0871,0.08,0.07466,0.0879,0.0613,0.05594,0.05596,0.038221,0.03464,0.01422],
    "Tumble_Drier": [0.004,0,0,0,0.00266,0.00177,0.01155,0.0231,0.0311,0.0391,0.0462,0.0524,0.0506,0.03,0.02,0.03,0.025,0.02,0.0453,0.0435,0.0426,0.0293,0.0204,0.0088],
    "Cooker":       [0.00259,0.00259,0.0103,0.01168,0.01168,0.02727,0.048,0.11,0.1221,0.111,0.1051,0.1233,0.148,0.05,0.04,0.03,0.08,0.1,0.15,0.1519,0.0883,0.0494,0.0337,0.01168],
    "Oven":         [0.00259,0.00519,0.00649,0.00259,0.00519,0.00519,0.0337,0.0688,0.0467,0.0337,0.03636,0.0415,0.0688,0.018,0.014,0.02,0.02,0.014,0.0987,0.0753,0.03636,0.0259,0.035,0.0324],
    "Grill":        [0,0,0.0012,0.0026,0.002597,0.00129,0.0012,0.0091,0.00779,0.00779,0.00259,0.01558,0.0519,0.01,0.002,0.001,0.025,0.03,0.0311,0.01558,0.0103,0.0011,0.00259,0.0026],
    "Hob":          [0.0025,0.0013,0.0026,0.00389,0.001298,0.00389,0.0155,0.0909,0.112,0.0688,0.061,0.0493,0.0727,0.01,0.03,0.02,0.04,0.08,0.13,0.089,0.0584,0.04025,0.01039,0.00779],
    "TV":           [0.0497,0.0135,0.01917,0.07123,0.042831,0.07345,0.1345,0.2,0.23,0.2228,0.17986,0.08254,0.12102,0.117024,0.0817,0.11195,0.168505,0.18,0.210705,0.36,0.35,0.31,0.3078,0.0904],
    "Electronics":  [0.02485,0.00675,0.009585,0.035615,0.0214155,0.036725,0.06725,0.1,0.115,0.1114,0.08993,0.04127,0.06051,0.058512,0.04085,0.055975,0.0842525,0.09,0.1053525,0.18,0.175,0.155,0.1539,0.0452],
}


#ALL appliance characteristics from Williams et al. (2025), refer to the Github link at the top of this file for source
#Power in kW and runtime in minutes are sampled once per agent and remain fixed from normal distributions with these parameters
#Reflects different households owning different appliance models with different efficiencies
#In the end, all are normalized (the agents pay per kw) but this will give a more reflective aggregate
#Max_uses_mu and max_uses_sigma added to cap physically unrealistic daily use count and add some heterogeneity

characteristics = {
    "Dishwasher": {"power_mu": 0.65, "power_sigma": 0.20, "runtime_mu": 60, "runtime_sigma": 15, "max_uses_mu": 1.0, "max_uses_sigma": 0.5},
    "Washing": {"power_mu": 0.65, "power_sigma": 0.40, "runtime_mu": 45, "runtime_sigma": 15, "max_uses_mu": 1.5, "max_uses_sigma": 0.5},
    "Tumble_Drier": {"power_mu": 1.10, "power_sigma": 0.70, "runtime_mu": 60, "runtime_sigma": 15, "max_uses_mu": 1.5, "max_uses_sigma": 0.5},
    "Cooker": {"power_mu": 1.00, "power_sigma": 0.80, "runtime_mu": 30, "runtime_sigma": 15, "max_uses_mu": 4.0, "max_uses_sigma": 0.5},
    "Oven": {"power_mu": 0.70, "power_sigma": 0.50, "runtime_mu": 30, "runtime_sigma": 15, "max_uses_mu": 2.0, "max_uses_sigma": 0.5},
    "Grill": {"power_mu": 1.50, "power_sigma": 0.50, "runtime_mu": 20, "runtime_sigma": 10, "max_uses_mu": 3.0, "max_uses_sigma": 1.0},
    "Hob": {"power_mu": 1.00, "power_sigma": 0.80, "runtime_mu": 20, "runtime_sigma": 10, "max_uses_mu": 4.0, "max_uses_sigma": 1.0},
    "TV": {"power_mu": 0.10, "power_sigma": 0.10, "runtime_mu": 150, "runtime_sigma": 60, "max_uses_mu": 6.0, "max_uses_sigma": 3.0},
    "Electronics": {"power_mu": 0.80, "power_sigma": 0.50, "runtime_mu": 30, "runtime_sigma": 60, "max_uses_mu": 8.0, "max_uses_sigma": 5.0},
}


def multi_peak_distribution(peak_list, baseline_probability=0.005):
    """
    Build a final 24-hour probability distribution as a sum of Gaussians

    Each entry in peak_list is a tuple representing a peak: (center_hour, height, width)
      - center_hour: on what hour the peak is located
      - height: peak magnitude before normalization
      - width: standard deviation in hours, if larger = broader bump

    A small flat baseline is added so every hour has at least some probability, to account for irregularities in schedule
    The result is then divided by the lists' sum for all elements so it adds up to 1.
    """
    distribution = np.zeros(24)  #initialize where the final distribution will be stored

    for center_hour, height, width in peak_list:
        #compute the Gaussian curve for this peak across all 24 hours
        gaussian = height * np.exp(-(np.arange(24, dtype=float) - center_hour) ** 2 / (2 * width ** 2))
        distribution = distribution + gaussian  #add this peak to the cumulative distribution

    distribution += baseline_probability  #add a small floor so no hour has zero probability
    distribution /= distribution.sum()    #normalize to a legal probability distribution which sums to 1
    return distribution

def adjust_peaks(peak_list):
    """
    The adjust_peaks function has been implemented to slightly shift all baseline distributions
    This change was made because of:
    1. A mismatch of peaking and minimal usage hours against real DSO data: without adjustments
       the model peaks at 12, with adjustments this peak is shifted to the morning and mirrors
       a similar structure to general electricity usage curves -> validated against https://huggingface.co/datasets/OpenSTEF/liander2024-energy-forecasting-benchmark
    2. This mismatch would make the usage of baseline EPEX pricing illogical
    """
    adjusted = []
    for center, height, width in peak_list:
        if 8 <= center <= 13:        #morning/midday peaks: shift earlier and sharpen
            center = center - 2.8    #pull toward morning hours
            height = height + 0.1    #slightly increase peak magnitude
            width = max(0.1, width - 0.3) #narrow peak peak, min width of 0.1 to avoid negative width
        if 18 <= center <= 22:       #evening peaks: shift earlier and broaden
            center = center - 2      #pull toward earlier evening
            height = height + 0.5    #increase peak magnitude  
            width = width + 0.5      #broaden to spread over more of the evening
        adjusted.append((center, height, width))
    return adjusted


#Raw fitted Gaussian tuples from Baseline_Distributions.py, copy-pasted from the print output
#These are the (uncalibrated) peaks of all appliances

dishwasher_raw_peaks = [(18.956315861064553, 0.11089493744353027, 1.357159179717805), (22.36027871623913, 0.07797395460554271, 0.771610889785124), (8.342960587362622, 0.05962600655372998, 1.4002245664011947), (13.314358999677186, 0.05, 1.8967520713194168)]
washing_raw_peaks = [(9.545167198482746, 0.7790387292870447, 2.0244500172858944), (15.60115790675859, 0.33413792870222453, 4.41870480568728)]
tumble_raw_peaks = [(11.62212340248307, 1.208933954615057, 1.417355530924213), (15.14387596374066, 0.5874527036199261, 0.6935500197957881), (8.466416528180892, 0.8074778473489305, 1.6964259012427485), (19.228974958574852, 1.1762361135932435, 2.0189138827026043)]
cooker_raw_peaks = [(18.153061228523544, 1.2582844975217007, 2.3746642890017986), (18.59097312234728, 0.6816961836975054, 0.6823292016136122), (11.795692959571095, 1.3953243916233937, 0.7497850471616758), (8.425787475465675, 1.4349642274396488, 2.0156280604983725)]
oven_raw_peaks = [(18.458235649470627, 0.37922201728501226, 0.3046711094792303), (11.738449852884226, 0.10562985711043978, 0.693090134001847), (7.545761516821308, 0.09437391474553104, 1.4517620706320356), (21.682585961077926, 0.05, 5.0)]
grill_raw_peaks = [(11.924944463813114, 0.2809556783426315, 0.5815431599228617), (7.95645163258788, 0.05, 1.1849438827408596), (17.781582463893, 0.16119629241571146, 1.2220586595698648), (16.406340924304615, 0.18506115113627022, 0.3062559743946323)]
hob_raw_peaks = [(18.22784853881954, 0.15979319165991923, 1.5625426291399946), (7.638187212842916, 0.14974437128118528, 0.8485970203176886), (10.844037221309907, 0.08465959403479242, 1.7520044963356904)]
tv_raw_peaks = [(8.178903464086682, 2.4674115102346237, 2.509022410606133), (19.505228878954412, 3.575007824823448, 0.4148362459235295), (18.615440531185566, 2.3700823150476538, 3.4375909799857913), (21.537457239174596, 3.6798085662182154, 0.4212647664517777)]
electronics_raw_peaks = [(8.178903464086682, 2.4674115102346237, 2.509022410606133), (19.505228878954412, 3.575007824823448, 0.4148362459235295), (18.615440531185566, 2.3700823150476538, 3.4375909799857913), (21.537457239174596, 3.6798085662182154, 0.4212647664517777)]

#Apply adjust_peaks() to produce the calibrated starting peaks for all agents
#These are the peaks before habit shift, agent.py adds its own height bonus on top
baseline_peak_tuples = {
    "Dishwasher": adjust_peaks(dishwasher_raw_peaks),
    "Washing": adjust_peaks(washing_raw_peaks),
    "Tumble_Drier": adjust_peaks(tumble_raw_peaks),
    "Cooker": adjust_peaks(cooker_raw_peaks),
    "Oven": adjust_peaks(oven_raw_peaks),
    "Grill": adjust_peaks(grill_raw_peaks),
    "Hob": adjust_peaks(hob_raw_peaks),
    "TV": adjust_peaks(tv_raw_peaks),
    "Electronics": adjust_peaks(electronics_raw_peaks)}

baselines = {}
#Iterate through appliances and their peak definitions
for name, peaks in baseline_peak_tuples.items():
    #generate the probability distribution for this specific appliance/task
    distribution = multi_peak_distribution(peaks)
    #store in the dictionary, appliance name as key
    baselines[name] = distribution
    
#EV baseline: manually added and peaks visually approximated from Robinson et al. (2013) figure 6: blue line called Home Private
#EVs are handled separately because they use a single daily draw rather than multiple uses
ev_raw_peaks = [(0.6, 2.2, 1.8), (14.0, 0.5, 3), (19.5, 2.5, 2.3), (24, 1, 2.7)]
ev_baseline  = multi_peak_distribution(adjust_peaks(ev_raw_peaks))


#---------------------
#Initialising an agent
#---------------------

def sample_agent_appliances(random_state):
    """
    Setting fixed characteristics for one agent per appliance
    Returns a dictionary with power + runtime for every agent if it has an ev 
    """
    agent_appliances = {}
    for name, chara in characteristics.items():
        power = abs(random_state.normal(chara["power_mu"], chara["power_sigma"]))  #abs as safety for extreme outliers
        runtime = abs(random_state.normal(chara["runtime_mu"], chara["runtime_sigma"]))
        runtime = max(1, round(runtime))  #again, safety for extreme outliers, must be at least 1 minute
        #max uses sampled per agent so households differ in how intensively they use each appliance
        max_uses = abs(random_state.normal(chara["max_uses_mu"], chara["max_uses_sigma"]))
        max_uses = max(1, round(max_uses))  #at least 1, must be integer
        agent_appliances[name] = {"power_kw": power, "runtime_min": runtime, "max_uses": max_uses}

    #baseline for standby, fridge etc:
    baseline_power = abs(random_state.normal(0.4, 0.15))  #sourced from Williams et al. (2025)
    agent_appliances["Baseline"] = {"power_kw": baseline_power}
    #this baseline won't matter for the costs, it is a tool to get a validated aggregate and to get more realistic peak loads

    #setting up EVs
    #EV characteristics from Robinson et al. (2013) Table 4
    has_ev = random_state.random() < 0.05  #5% EV ownership is a hyperparameter based on dutch data interpolated to 2024 (https://www.rvo.nl/onderwerpen/elektrisch-vervoer/stand-van-zaken), to be adjusted when addressing specific countries
    if has_ev:
        ev_power = abs(random_state.normal(3.3, 0.3))      #3 kv mentioned in Robinson et al. (2013), converted to 3.3 kw
        ev_runtime = abs(random_state.normal(3.1 * 60, 20))  #3.1 hours converted to minutes, 20 is a self set hyperparameter to account for variance in batteries/car-types
        ev_runtime = max(15, round(ev_runtime))  #safety for edge cases, minimum 15 minutes of charging
        agent_appliances["EV"] = {"power_kw": ev_power, "runtime_min": ev_runtime}
    return agent_appliances, has_ev


#----------------------------------------------
#Building the daily usage (amount and timeslot)
#----------------------------------------------

def build_daily_load(agent_appliances, has_ev, random_state, previous_overflow=None, custom_distributions=None):
    """
    Generate one day electricity load profile for single agent

    Overnight overflow from the previous day is added at the top so that
    appliances that run over midnight on DAY-1 contribute to DAY

    Parameters:
    -> agent_appliances: dict from sample_agent_appliances()
    -> has_ev: True if possesses electric vehicle
    -> random_state
    -> previous_overflow: slot list with overflow load carried over from the previous day, or none
    -> custom_distributions: dict with mpd for appliances or None
        -> If provided by the shifting algorithm in agent.py, each appliance samples start-hours
           from this agent-specific shifted distribution rather than the global baseline, only used in first iter
 
    Returns:
    -> load profile: kW per 15-min slot for today 
    -> overflow: kW per 15-min slot spilling to tomorrow 
    -> schedule: dict with {appliance_name: [start_slot, (for all times that agents in starting that appliance]}
    """
    load_profile = np.zeros(96) #24h x 4 for 15 min intervals
    overflow = np.zeros(96) #load that spills past midnight and carries to tomorrow
    schedule = {} #records which start slots each appliance will use today

    #add day-1 overflow to today  
    if previous_overflow is not None:
        for slot in range(96):
            load_profile[slot] = load_profile[slot] + previous_overflow[slot]

    #spread baseline across every slot of the day
    baseline_power = agent_appliances["Baseline"]["power_kw"]
    for slot in range(96):
        load_profile[slot] = load_profile[slot] + baseline_power

    #pre-build the active distributions dict once before the per-appliance loop
    #if custom_distributions were provided by agent.py, they override the shared baselines
    #appliances not in custom_distributions still fall back to the global baseline
    if custom_distributions is not None:
        active_distributions = baselines.copy() #start with the shared baselines
        active_distributions.update(custom_distributions) #overlay with the agent's shifted versions
    else:
        active_distributions = baselines  #no shifting active, use shared baselines as is

    #now for normal appliances
    for name, probs in data.items():  #iterate over all switch-on data, if it hits, n+1 usage
        n_uses = 0
        for p in probs:
            draw = random_state.random()  #draw a random float between 0 and 1
            if draw < p:
                n_uses = n_uses + 1  #this hour produced a switch-on event

        #cap n_uses at the agent's sampled max_uses for this appliance
        #prevents unrealistic use counts (e.g. TV 8 times), as determined by Williams et al. (2025)
        n_uses = min(n_uses, agent_appliances[name]["max_uses"])

        #skip this appliance if it will not be used today
        if n_uses == 0:
            schedule[name] = []
            continue

        #now sample which hour to take from the active distribution for this appliance
        dist = active_distributions[name].copy()  
        start_hours = random_state.choice(np.arange(24), size=n_uses, p=dist)

        #Let the agent sample 5 times again for new hours, if an hour was already chosen
        #to prevent a lot of duplicate use and create more logical, spread-over-the-day behavior
        used_hours = []
        for hour in start_hours:
            for i in range(5):
                if hour not in used_hours:
                    used_hours.append(hour)  #accept this hour, it has not been used yet
                    break
                hour = random_state.choice(np.arange(24), p=dist)  #resample and try again
            else:
                #after 5 failed attempts, take the value anyway
                used_hours.append(hour)

        #get this agent's fixed characteristic values for this appliance
        power_kw = agent_appliances[name]["power_kw"]
        runtime_min = agent_appliances[name]["runtime_min"]
        n_slots = max(1, round(runtime_min / 15))  #convert runtime from minutes to 15-min slots

        start_slots = []
        for h in used_hours:
            #to determine the precise 15-minute start slot,
            #pick a random quarter-hour within the sampled hour
            quarter = int(random_state.integers(0, 4))
            start_slot = h * 4 + quarter

            #spread load over each slot of the runtime
            #if the slot exceeds 95 (end of day), place load in the overflow array
            for slot in range(n_slots):
                target_slot = start_slot + slot
                if target_slot < 96:
                    #slot is within today, add to today's load profile
                    load_profile[target_slot] = load_profile[target_slot] + power_kw
                else:
                    #slot goes past midnight, add to overflow which carries to tomorrow
                    overflow_slot = target_slot - 96
                    overflow[overflow_slot] = overflow[overflow_slot] + power_kw
            start_slots.append(start_slot)

        schedule[name] = start_slots  #record the start slots for this appliance today

    #handling EVs
    #if the probability hits, one start-time is sampled from the EV MPD,
    #and load is spread over the charging duration exactly as done above
    if has_ev:
        ev_draw = random_state.random()
        if ev_draw < 41.6 / 180:  #taken from table 4 (private home value divided by 6 months)
            #sample start-hour from the EV evening-peak distribution
            start_hour = int(random_state.choice(np.arange(24), p=ev_baseline))
            quarter = int(random_state.integers(0, 4))
            start_slot = start_hour * 4 + quarter  #convert to 15-min slot index

            power_kw = agent_appliances["EV"]["power_kw"]
            runtime_min = agent_appliances["EV"]["runtime_min"]
            n_slots = max(1, round(runtime_min / 15))

            ev_slots = []
            #same procedure as above with overflow and spread
            for s in range(n_slots):
                target_slot = start_slot + s

                if target_slot < 96:
                    load_profile[target_slot] = load_profile[target_slot] + power_kw
                else:
                    overflow_slot = target_slot - 96
                    if overflow_slot < 96:
                        overflow[overflow_slot] = overflow[overflow_slot] + power_kw

                ev_slots.append(start_slot + s)
            schedule["EV"] = ev_slots
        else:
            schedule["EV"] = []  #no charging today beacuse probability did not hit

    return load_profile, overflow, schedule