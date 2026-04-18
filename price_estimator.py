"""
price_estimator.py — Hourly electricity price estimation based on simulated demand.

Combines three data sources all computed in the price_model_baselines folder:
    -> solar_elasticity : hour-level sensitivity of price to demand
        -> Low during solar peak (around noon) because surplus supply keeps prices stable
        -> High in the evening and night when solar is absent and grid is stressed
    -> expected_demand : per-agent average hourly demand (kW) across 50 simulation seeds
        -> Acts as the "neutral" reference: when actual demand equals expected, no price change
    -> price_baseline : mean EPEX NL hourly prices divided by 10 for cleaner numbers
        -> When demand equals expected exactly, estimated price equals price_baseline

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

#Mean solar elasticity per hour of day, computed from 5 Liander solar park measurement files
#Low values around noon (solar surplus), high values at night (no solar supply)
#Sourced from price_model_baselines/solar_elasticity.py
solar_elasticity = [
    0.5985212640633001,   #hour 0  (midnight, high elasticity)
    0.5985255158965563,   #hour 1
    0.5985256676662337,   #hour 2
    0.598337080905063,    #hour 3
    0.5919722408756971,   #hour 4
    0.561719681233165,    #hour 5
    0.49254671652996,     #hour 6  (dawn, elasticity starting to fall)
    0.3897513351009402,   #hour 7
    0.27386045210681276,  #hour 8
    0.17804050319985892,  #hour 9
    0.12026228481734069,  #hour 10
    0.09999999999999998,  #hour 11 (near-minimum, most solar supply)
    0.10779434049122755,  #hour 12
    0.14652344967549197,  #hour 13
    0.219455068428624,    #hour 14
    0.31772045265215004,  #hour 15
    0.42432822155873384,  #hour 16
    0.5164913025719808,   #hour 17 (solar fading, elasticity climbing)
    0.5717617136381481,   #hour 18
    0.5941506085544515,   #hour 19
    0.5983933799651381,   #hour 20
    0.5985306872264536,   #hour 21
    0.5985270139786539,   #hour 22
    0.5985216742651864,   #hour 23
]

#Expected (baseline) per-agent hourly demand in kW
#Computed in price_model_baselines/usage_baseline.py across 50 seeds x 5 days x 500 agents
#Used as the reference point: when actual simulated demand equals these values, price = price_baseline
expected_demand = [
    0.43953643,  #hour 0  (low demand at night)
    0.42574971,  #hour 1
    0.43836126,  #hour 2
    0.47233758,  #hour 3
    0.53141863,  #hour 4
    0.60753253,  #hour 5
    0.66313066,  #hour 6  (morning ramp-up)
    0.67250915,  #hour 7
    0.6465164,   #hour 8
    0.66876732,  #hour 9
    0.5535738,   #hour 10
    0.50941658,  #hour 11
    0.52060918,  #hour 12
    0.56453714,  #hour 13
    0.63382294,  #hour 14
    0.72724222,  #hour 15
    0.84408718,  #hour 16
    0.91658533,  #hour 17 (evening peak)
    0.91394429,  #hour 18
    0.85904491,  #hour 19
    0.77586794,  #hour 20
    0.6689581,   #hour 21
    0.57338553,  #hour 22
    0.50967155,  #hour 23
]

#Mean EPEX NL prices per hour in 2024, divided by 10 for smaller numbers
#Sourced from price_model_baselines/EPEX_baselines.py using the OpenSTEF Liander dataset
#These are the prices returned when demand exactly matches expected_demand
price_baseline = [
    6.845,   #hour 0
    6.583,   #hour 1
    6.439,   #hour 2  (cheapest overnight hours)
    6.666,   #hour 3
    7.703,   #hour 4
    8.799,   #hour 5
    9.07,    #hour 6
    8.434,   #hour 7
    7.218,   #hour 8
    6.05,    #hour 9
    5.074,   #hour 10
    4.374,   #hour 11 (midday solar surplus, cheapest daytime)
    4.267,   #hour 12
    4.962,   #hour 13
    6.341,   #hour 14
    8.422,   #hour 15
    10.84,   #hour 16
    12.351,  #hour 17 (evening peak, most expensive)
    12.241,  #hour 18
    10.409,  #hour 19
    9.121,   #hour 20
    8.299,   #hour 21
    7.767,   #hour 22
    7.219,   #hour 23
]


def hour_price_estimator(sim_demand_yesterday):
    """
    Estimate hourly electricity prices for today based on yesterday's per-agent demand.

    Parameters:
    - sim_demand_yesterday: list or array of 24 floats
        -> Mean per-agent hourly demand in kW from the previous simulated day
        -> Must be in the same units as expected_demand (kW per agent per hour)
        -> Computed in run_model.py as: aggregate.reshape(24,4).mean(axis=1) / N

    Returns a list of 24 floats representing estimated price per hour in pricing units
    """
    hour_prices = []  #will hold the final 24 price estimates

    for hour in range(0, 24):
        #compute the relative deviation of simulated demand from the expected baseline
        #if demand is higher than expected, the fraction is positive -> price goes up
        #if demand is lower than expected, the fraction is negative -> price goes down
        demand_deviation = (sim_demand_yesterday[hour] - expected_demand[hour]) / expected_demand[hour]

        #scale the deviation by solar elasticity and apply to the baseline price
        #solar_elasticity[hour] controls how much the deviation actually moves the price
        price_h = price_baseline[hour] * (1 + solar_elasticity[hour] * demand_deviation)

        hour_prices.append(round(price_h, 3))  #round to 3 decimals for cleanliness

    return hour_prices


#-------------------------
#Standalone test
#Not executed when imported by run_model.py, only runs if this file is called directly
#-------------------------

if __name__ == "__main__":
    #Feed back expected_demand with a small increase to test that prices rise accordingly
    test_demand = [h + 0.1 for h in expected_demand]
    result = hour_price_estimator(test_demand)
    print("Test prices (slight demand increase above expected):", result)