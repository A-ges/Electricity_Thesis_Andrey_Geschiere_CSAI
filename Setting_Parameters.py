import pandas as pd
import numpy as np

#Fixed behavioral parameter configurations per agent type
#High mu = generally higher score, high K = high variance
beta_parameters = {
    "Habit-driven": {
        "habit":  {"mu": 0.85, "K": 10},
        "price":  {"mu": 0.10, "K": 5},
        "social": {"mu": 0.15, "K": 5}},
    "Price-responsive": {
        "habit":  {"mu": 0.50, "K": 30},
        "price":  {"mu": 0.85, "K": 10},
        "social": {"mu": 0.10, "K": 5}},
    "Social-influenced": {
        "habit":  {"mu": 0.70, "K": 20},
        "price":  {"mu": 0.10, "K": 5},
        "social": {"mu": 0.85, "K": 10}}
}
#main group variable should be high in any group
#------------HABIT-------------
#Habit agents have a lower price responsivity in general, because of unknowingness or lack of interest in self induced shift
#Habit agents have a higher social influence parameter in general, allowing them to somewhat pivot towards a better strategy in case their network incentivises them. This follows the Cultural Attractor Theory as mentioned in Falandays & Smaldino (2022) 

#------------PRICE-------------
#Price responsive agents have some habit, but are allowed to spread more to indicate their stance towards flexibility
#They have the lowest social parameter of all, because these agents have already set their strategy. They are affected to some extent but don't have a superior strategy to pivot towards, unlike habitual agents

#------------SOCIAL-------------
#These agents are viewed as initially habitual agents, who are generally more open to change a new strategy as long as their network shows this behavior.
#Therefore, they start with a generally higher, but more spread (compared to a true habit agent), habit parameter.
#Their initial knowledge about price is the same as the original habit-driven agents, allowing them this pivot oppurtunity through their social parameter


def Param_Init(habit_num, price_num, social_num, random_state=None):
    """
    Initialize a population of agents with behavioral parameters sampled
    from group-specific Beta distributions as seen above

    Parameters:
    -> habit_num: Number of Habit-driven agents
    -> price_num: Number of Price-responsive agents
    -> social_num: Number of Social-influenced agents
    -> random_state

    Returns a Pandas df with one row per agent with columns [dominant_group | habit_str | price_sens | soc_suc]
    """
    rng = np.random.default_rng(random_state) #setting the same random seed for all random decisions in this file

    mix = {
        "Habit-driven": habit_num,
        "Price-responsive": price_num,
        "Social-influenced": social_num,
    }

    all_agents = []

    for group_name, count in mix.items():
        group_cfg = beta_parameters[group_name]

        for i in range(count):   #loop to draw from all beta distributions for every agent
                                 #Uses np.random.beta for draws, refer to https://numpy.org/doc/stable/reference/random/generated/numpy.random.beta.html
            agent = {
                "dominant_group": group_name,

                "habit_str": rng.beta(group_cfg['habit']['mu'] * group_cfg['habit']['K'],
                    (1 - group_cfg['habit']['mu']) * group_cfg['habit']['K']),

                "price_sens": rng.beta(group_cfg['price']['mu'] * group_cfg['price']['K'],
                    (1 - group_cfg['price']['mu']) * group_cfg['price']['K']),

                "soc_suc": rng.beta(
                    group_cfg['social']['mu'] * group_cfg['social']['K'],
                    (1 - group_cfg['social']['mu']) * group_cfg['social']['K'])
            }
            all_agents.append(agent)

    return pd.DataFrame(all_agents)
