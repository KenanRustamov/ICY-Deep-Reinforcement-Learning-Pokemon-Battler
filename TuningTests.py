def trainingTuner(model, n_action, policy, memory, trainEnv, dqnDict, randomAgent, maxAgent, heuristicsAgent, maxLen):
    steps = 10000
    for i in range(1, maxLen + 1):
        for j in range(0, maxLen + 1):
            for k in range(0, maxLen + 1):
                dqn = DQNAgent(
                    model=model,
                    nb_actions=n_action,
                    policy=policy,
                    memory=memory,
                    nb_steps_warmup=1000,
                    gamma=0.5,
                    target_model_update=1,
                    delta_clip=0.01,
                    enable_double_dqn=True,
                )
                dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])

                trainAgainstAgent(dqn, i*steps, trainEnv, randomAgent)
                if j: trainAgainstAgent(dqn, j*steps,trainEnv, maxAgent, True)
                if k: trainAgainstAgent(dqn, k*steps, trainEnv, heuristicsAgent, True)

                dqnDict[(i*steps,j*steps,k*steps)] = dqn
                if (not (i == maxLen and j == maxLen and k == maxLen)): trainEnv.reset_env(restart = True, opponent = RandomPlayer(battle_format="gen8randombattle"))