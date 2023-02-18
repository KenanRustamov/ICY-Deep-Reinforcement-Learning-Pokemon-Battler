import asyncio
import numpy as np

from gym.spaces import Space, Box
from gym.utils.env_checker import check_env
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from tabulate import tabulate
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Flatten, Normalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player import (
    background_evaluate_player,
    background_cross_evaluate,
    Gen8EnvSinglePlayer,
    RandomPlayer,
    MaxBasePowerPlayer,
    ObservationType,
    wrap_for_old_gym_api,
    SimpleHeuristicsPlayer,
)


class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle,fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def embed_battle(self, battle: AbstractBattle) -> ObservationType:
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        can_dynamax = np.ones(1)
        team_type = np.zeros(12)
        opponent_team_type = np.zeros(12)
        team_multiplyer = -np.ones(6)
        dynamax_turn = np.ones(1)
        moves_status_effects = np.zeros(4)
        current_weather = np.zeros(1)
        opponent_dynamax_turn = np.ones(1)
        opponent_can_dynamax = np.ones(1)
        opponent_side_conditions = np.zeros(20)
        team_health = np.zeros(6)
        opponent_team_health = np.zeros(6)
        active_fields = np.zeros(12)
        active_pokemon_side_conditions = np.zeros(20)
        active_opponent_status = np.zeros(1)
        active_pokemon_status = np.zeros(1)

        can_dynamax[0] = 1 if battle.can_dynamax else 0
        dynamax_turn[0] = battle.dynamax_turns_left/3 if battle.dynamax_turns_left != None else -1
        opponent_dynamax_turn[0] = battle.opponent_dynamax_turns_left/3 if battle.opponent_dynamax_turns_left != None else -1
        current_weather[0] = 0 if len(battle.weather) == 0 else list(battle.weather.items())[0][0].value/8
        opponent_can_dynamax[0] = 1 if battle._opponent_can_dynamax else 0
        active_opponent_status[0] = battle.opponent_active_pokemon.status.value/6 if battle.opponent_active_pokemon.status else 0
        active_pokemon_status[0] = battle.active_pokemon.status.value/6 if battle.active_pokemon.status else 0

        # for effect, val in battle.opponent_active_pokemon.effects().items():
            # print(effect)

        for field,turn in battle.fields.items():
            active_fields[field.value - 1] = 1

        for sideCondition,val in battle.opponent_side_conditions.items():
            opponent_side_conditions[sideCondition.value - 1] = 1
        
        for sideCondition,val in battle.side_conditions.items():
            active_pokemon_side_conditions[sideCondition.value - 1] = 1

        for i,pokemon in enumerate(battle.available_switches):
            firstTypeMultiplyer = pokemon.type_1.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,)
            team_multiplyer[i] = firstTypeMultiplyer

            if pokemon.type_2 != None:
                secondTypeMultiplyer = pokemon.type_2.damage_multiplier(
                        battle.opponent_active_pokemon.type_1,
                        battle.opponent_active_pokemon.type_2,)
                team_multiplyer[i] *= secondTypeMultiplyer
            team_multiplyer[i] /= 4

        for i,pokemon in enumerate(battle.team.values()):
            i = i*2
            if pokemon.fainted:
                team_type[i] = 0
                team_type[i + 1] = 0
            else:
                team_type[i] = pokemon.type_1.value/19 if pokemon.type_1 != None else 0
                team_type[i + 1] =  pokemon.type_2.value/19 if pokemon.type_2 != None else 0
                team_health[i//2] = pokemon.current_hp/800 if pokemon.current_hp else 0 #divide by maximum possible HP


        for i, pokemon in enumerate(battle.opponent_team.values()):
            i = i*2
            if pokemon.fainted:
                team_type[i] = 0
                team_type[i + 1] = 0
            else:
                opponent_team_type[i] = pokemon.type_1.value/19 if pokemon.type_1 != None else 0
                opponent_team_type[i + 1] =  pokemon.type_2.value/19 if pokemon.type_2 != None else 0
                opponent_team_health[i//2] = pokemon.current_hp/800 if pokemon.current_hp else 0#divide by maximum possible HP

        for i, move in enumerate(battle.available_moves):
            moves_status_effects[i] = move.status.value/7 if move.status != None else 0
            moves_base_power[i] = (
                move.base_power / 300
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,) / 4

        # We count how many pokemons have fainted in each team
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6)

        # Final vector with 12 components
        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
                can_dynamax,
                dynamax_turn,
                team_type,
                opponent_team_type,
                team_multiplyer,
                moves_status_effects,
                current_weather,
                opponent_dynamax_turn,
                opponent_can_dynamax,
                opponent_side_conditions,
                team_health,
                opponent_team_health,
                active_fields,
                active_pokemon_side_conditions,
                active_opponent_status,
                active_pokemon_status
            ]
        )
        return np.float32(final_vector)

    def describe_embedding(self) -> Space:
        low = []
        moveBasePowerLower = [-1]*4
        moveDamageMultiplyerLower = [0]*4
        faintedTeamLower = [0]
        faintedOpponentTeamLower = [0]
        canDynamaxLower = [0]
        dynamaxTurnLower = [-1]
        teamTypeLower= [0]*12
        opponentTeamTypeLower = [0]*12
        teamMultiplyerLower = [-1]*6
        moveStatusLower = [0]*4
        currentWeatherLower = [0]
        opponentDynamaxTurnLower = [-1]
        opponentCanDynamaxLower = [0]
        opponentSideConditionsLower = [0]*20
        teamHealthLower = [0]*6
        opponentTeamHealthLower = [0]*6
        activeFieldsLower = [0]*12
        activePokemonSideConditionsLower = [0]*20
        activeOpponentStatusLower = [0]
        activePokemonStatusLower = [0]

        low += moveBasePowerLower
        low += moveDamageMultiplyerLower
        low += faintedTeamLower
        low += faintedOpponentTeamLower
        low += canDynamaxLower
        low += dynamaxTurnLower
        low += teamTypeLower
        low += opponentTeamTypeLower
        low += teamMultiplyerLower
        low += moveStatusLower
        low += currentWeatherLower
        low += opponentDynamaxTurnLower
        low += opponentCanDynamaxLower
        low += opponentSideConditionsLower
        low += teamHealthLower
        low += opponentTeamHealthLower
        low += activeFieldsLower
        low += activePokemonSideConditionsLower
        low += activeOpponentStatusLower
        low += activePokemonStatusLower
        

        high = []
        moveBasePowerUpper = [1]*4
        moveDamageMultiplyerUpper = [1]*4
        faintedTeamUpper = [1]
        faintedOpponentTeamUpper = [1]
        canDynamaxUpper = [1]
        dynamaxTurnUpper = [1]
        teamTypeUpper = [1]*12
        opponentTeamTypeUpper = [1]*12
        teamMultiplyerUpper = [1]*6
        moveStatusUpper = [1]*4
        currentWeatherUpper = [1]
        opponentDynamaxTurnUpper = [1]
        opponentCanDynamaxUpper = [1]
        opponentSideConditionsUpper = [1]*20
        teamHealthUpper = [1]*6
        opponentTeamHealthUpper = [1]*6
        activeFieldsUpper = [0]*12
        activePokemonSideConditionsUpper = [1]*20
        activeOpponentStatusUpper = [1]
        activePokemonStatusUpper = [1]
        
        high += moveBasePowerUpper
        high += moveDamageMultiplyerUpper
        high += faintedTeamUpper
        high += faintedOpponentTeamUpper
        high += canDynamaxUpper
        high += dynamaxTurnUpper
        high += teamTypeUpper
        high += opponentTeamTypeUpper
        high += teamMultiplyerUpper
        high += moveStatusUpper
        high += currentWeatherUpper
        high += opponentDynamaxTurnUpper
        high += opponentCanDynamaxUpper
        high += opponentSideConditionsUpper
        high += teamHealthUpper
        high += opponentTeamHealthUpper
        high += activeFieldsUpper
        high += activePokemonSideConditionsUpper
        high += activeOpponentStatusUpper
        high += activePokemonStatusUpper

        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )
def buildModelLayers(model,inputShape, outputLen):
    model.add(Dense(inputShape[1], activation="elu", input_shape=inputShape))
    model.add(Normalization())
    model.add(Flatten())
    model.add(Dense(32, activation="relu"))
    model.add(Dense(outputLen, activation="linear"))

def restartAndTrainRandom(dqn, steps, trainingEnv):
    trainingEnv.reset_env(restart=True, opponent=RandomPlayer(battle_format="gen8randombattle"))
    dqn.fit(trainingEnv, nb_steps=steps)

def restartAndTrainMaxDamage(dqn, steps, trainingEnv):
    trainingEnv.reset_env(restart=True, opponent=MaxBasePowerPlayer(battle_format="gen8randombattle"))
    dqn.fit(trainingEnv, nb_steps=steps)

def restartAndTrainHeuristic(dqn, steps, trainingEnv):
    trainingEnv.reset_env(restart=True, opponent=SimpleHeuristicsPlayer(battle_format="gen8randombattle"))
    dqn.fit(trainingEnv, nb_steps=steps)


async def main():
    # First test the environment to ensure the class is consistent
    # with the OpenAI API
    opponent = RandomPlayer(battle_format="gen8randombattle")
    test_env = SimpleRLPlayer(battle_format="gen8randombattle", start_challenging=True, opponent=opponent)
    # test_env = wrap_for_old_gym_api(test_env)
    check_env(test_env)
    test_env.close()
    print("Test Environment Closed")

    # Create one environment for training and one for evaluation
    opponent = RandomPlayer(battle_format="gen8randombattle")
    second_opponent = MaxBasePowerPlayer(battle_format="gen8randombattle")
    third_opponent = SimpleHeuristicsPlayer(battle_format="gen8randombattle")
    train_env = SimpleRLPlayer(battle_format="gen8randombattle", opponent=opponent, start_challenging=True)
    train_env = wrap_for_old_gym_api(train_env)

    eval_env = SimpleRLPlayer(
        battle_format="gen8randombattle", opponent=third_opponent, start_challenging=True
    )
    eval_env = wrap_for_old_gym_api(eval_env)

    # Compute dimensions
    n_action = train_env.action_space.n
    input_shape = (1,) + train_env.observation_space.shape
    print(input_shape)

    # Create model
    model = Sequential()
    model.add(Dense(input_shape[1], activation="elu", input_shape=input_shape))
    model.add(Normalization())
    model.add(Flatten())
    model.add(Dense((input_shape[1] + n_action)//2, activation="elu"))
    model.add(Dense(n_action, activation="linear"))

    # Defining the DQN
    memory = SequentialMemory(limit=10000, window_length=1)

    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0.0,
        nb_steps=10000,
    )

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

    # Training the model
    dqn.fit(train_env, nb_steps=40000)

    # restartAndTrainMaxDamage(dqn, 40000,train_env)
    # restartAndTrainHeuristic(dqn, 10000, train_env)
    # restartAndTrainMaxDamage(dqn, 10000,train_env)

    train_env.close()

    # Evaluating the model
    print("Results against heuristic player:")
    dqn.test(eval_env, nb_episodes=100, verbose=False, visualize=False)
    print(
        f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    )
    eval_env.reset_env(restart=False)

    # print("Results against random player:")
    # dqn.test(eval_env, nb_episodes=100, verbose=False, visualize=False)
    # print(
    #     f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    # )
    # eval_env.reset_env(restart=True, opponent=second_opponent)

    # print("Results against max base power player:")
    # dqn.test(eval_env, nb_episodes=100, verbose=False, visualize=False)
    # print(
    #     f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    # )
    # eval_env.reset_env(restart=False)

    # Evaluate the player with included util method
    n_challenges = 250
    placement_battles = 40
    eval_task = background_evaluate_player(
        eval_env.agent, n_challenges, placement_battles
    )
    dqn.test(eval_env, nb_episodes=n_challenges, verbose=False, visualize=False)
    print("Evaluation with included method:", eval_task.result())
    eval_env.reset_env(restart=False)


    # Cross evaluate the player with included util method
    n_challenges = 50
    players = [
        eval_env.agent,
        RandomPlayer(battle_format="gen8randombattle"),
        MaxBasePowerPlayer(battle_format="gen8randombattle"),
        SimpleHeuristicsPlayer(battle_format="gen8randombattle"),
    ]
    cross_eval_task = background_cross_evaluate(players, n_challenges)
    dqn.test(
        eval_env,
        nb_episodes=n_challenges * (len(players) - 1),
        verbose=False,
        visualize=False,
    )
    cross_evaluation = cross_eval_task.result()
    table = [["-"] + [p.username for p in players]]
    for p_1, results in cross_evaluation.items():
        table.append([p_1] + [cross_evaluation[p_1][p_2] for p_2 in results])
    print("Cross evaluation of DQN with baselines:")
    print(tabulate(table))
    eval_env.close()
    dqn.save_weights("Saved Models/currentModel")


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
