import cityflow

if __name__ == '__main__':
    cfg_file = "cityflow_configs/manhattan_no_rl.json"
    N_STEPS = 3600

    eng = cityflow.Engine(cfg_file)
    travel_times = []

    for i_step in range(N_STEPS):
        eng.next_step()
        average_travel_time = eng.get_average_travel_time()
        travel_times.append(average_travel_time)
        print(f"\r i: {i_step}, avg travel time: " + str(average_travel_time), end="")

    print("\nFinal average travel time:", travel_times[-1])
