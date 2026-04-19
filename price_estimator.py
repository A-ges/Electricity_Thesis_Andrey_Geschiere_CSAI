"""
price_estimator.py — Hourly electricity price estimation based on simulated demand.

Combines three data sources all computed in the groundwork folder:
    -> solar_elasticity: hour-level sensitivity of price to demand
        -> Low during solar peak (around noon) because surplus supply keeps prices stable
        -> High in the evening and night when solar is absent and grid is stressed
    -> expected_demand: per-agent average hourly demand across 50 simulation seeds
        -> Acts as the "neutral" reference: when actual demand equals expected, no price change
    -> price_baseline: mean EPEX NL hourly prices from 2024 divided by 10 for cleaner numbers
        -> When simulated demand == expected demand, estimated price == price_baseline

Price formula (applied per hour):
    price(h) = price_baseline(h) * (1 + solar_elasticity(h) * (simulated_demand(h) - expected_demand(h)) / expected_demand(h))

    -> The middle factor measures the relative demand deviation at each hour
        -> Positive deviation (more demand than expected) -> price increases
        -> Negative deviation (less demand than expected) -> price decreases
    -> The solar_elasticity term scales the sensitivity
        -> At noon (low elasticity ≈ 0.1) a big demand change barely moves the price
        -> At midnight (high elasticity ≈ 0.6) the same demand change moves the price much more

NOTE: prices are in unnamed pricing units (EPEX €/MWh divided by 10)
-> They are only used for relative comparison within this model
-> They do not directly represent what a Dutch consumer pays
"""

#Mean solar elasticity per hour of day, computed from the 5 Liander solar park measurement files
#Low values around noon (solar surplus), high values at night (no solar supply)
#Sourced from groundwork/solar_elasticity.py
solar_elasticity = [0.5985212640633001, 0.5985255158965563, 0.5985256676662337, 0.598337080905063, 0.5919722408756971, 0.561719681233165, 0.49254671662996, 0.3897513351009402, 0.27386045210681276, 0.17804050319985892, 0.12026228481734069, 0.09999999999999998, 0.10779434049122755, 0.14652344967549197, 0.219455068428624, 0.31772045265215004, 0.42432822155873384, 0.5164913025719808, 0.5717617136381481, 0.5941506085544515, 0.5983933799651381, 0.5985306872264536, 0.5985270139786539, 0.5985216742651864]

#Expected per-agent hourly demand in kW, gotten from baseline
#Computed in groundwork/run_baseline.py across 50 seeds x 5 days x 500 agents
expected_demand = [0.43953643, 0.42574971, 0.43836126, 0.47233758, 0.53141863, 0.60753253, 0.66313066, 0.67250915, 0.6465164, 0.66876732, 0.5535738, 0.50941658, 0.52060918, 0.56453714, 0.63382294, 0.72724222, 0.84408718, 0.91658533, 0.91394429, 0.85904491, 0.77586794, 0.6689581, 0.57338553, 0.50967155]

#Mean EPEX NL prices per hour in 2024, divided by 10 for smaller numbers
#Sourced from price_model_baselines/EPEX_baselines.py using the OpenSTEF Liander dataset
#These are the prices returned when demand exactly matches expected_demand
price_baseline = [6.845, 6.583, 6.439, 6.666, 7.703, 8.799, 9.07, 8.434, 7.218, 6.05, 5.074, 4.374, 4.267, 4.962, 6.341, 8.422, 10.84, 12.351, 12.241, 10.409, 9.121, 8.299, 7.767, 7.219]

def hour_price_estimator(sim_demand_yesterday):
    """
    Estimate hourly electricity prices for today based on yesterday's per agent demand.

    Takes mean per-agent hourly demand in kW from the previous simulated day
    Returns estimated price per hour in same units as baseline
    """
    hour_prices = []   

    for hour in range(0, 24):
        #compute the relative deviation of simulated demand from the expected baseline
        #if demand is higher than expected, the fraction is positive -> price goes up
        #if demand is lower than expected, the fraction is negative -> price goes down
        demand_deviation = (sim_demand_yesterday[hour] - expected_demand[hour]) / expected_demand[hour]

        #main formula
        price_h = price_baseline[hour] * (1 + solar_elasticity[hour] * demand_deviation)
        hour_prices.append(round(price_h, 3))  
    return hour_prices
