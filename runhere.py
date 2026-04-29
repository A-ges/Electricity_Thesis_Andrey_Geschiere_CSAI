from run_model import run_model

df_agents, df_daily, load_profiles, df_pricing = run_model(
    agents_pct   = [50, 25, 25],
    # 50% habit-driven, 25% price, 25% social
    network_code = "300c",          #300-agent network, variant c
    days         = 30,
    graphs       = [1, 2],          #print aggregate load plots for days 1 and 2
    median_plot  = True,
    random_state = 40,
    epsilon_social = 0.5,
    epsilon_price = 0.25, 
    epsilon_habit = 1)

