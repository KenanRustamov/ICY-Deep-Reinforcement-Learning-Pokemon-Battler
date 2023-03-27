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
# from matplotlib import pyplot as plt

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
        activePokemon = battle.active_pokemon
        opponentActivePokemon = battle.opponent_active_pokemon

        activePokemonMovesBasePower = -np.ones(4)
        activePokemonMovesDmgMultiplier = -np.ones(4)
        canDynamax = np.ones(1)
        teamTypes = np.zeros(12)
        opponentTeamTypes = np.zeros(12)
        teamDmgMultiplyer = -np.ones(6)
        dynamaxTurn = np.ones(1)
        activePokemonMovesStatusEffects = -np.ones(4)
        currentWeather = np.zeros(1)
        opponentDyanamaxTurn = np.ones(1)
        opponentCanDynamax = np.ones(1)
        opponentSideConditions = np.zeros(20)
        teamHealth = np.zeros(6)
        opponentTeamHealth = np.zeros(6)
        activeFields = np.zeros(12)
        activePokemonSideConditions = np.zeros(20)
        activeOpponentPokemonStatus = np.zeros(1)
        activePokemonStatus = np.zeros(1)
        activePokemonStats = -np.ones(6)

        canDynamax[0] = 1 if battle.can_dynamax else 0
        dynamaxTurn[0] = battle.dynamax_turns_left/3 if battle.dynamax_turns_left != None else -1
        opponentDyanamaxTurn[0] = battle.opponent_dynamax_turns_left/3 if battle.opponent_dynamax_turns_left != None else -1
        currentWeather[0] = 0 if len(battle.weather) == 0 else list(battle.weather.items())[0][0].value/8
        opponentCanDynamax[0] = 1 if battle.opponent_can_dynamax else 0
        activeOpponentPokemonStatus[0] = opponentActivePokemon.status.value/6 if opponentActivePokemon.status else 0
        activePokemonStatus[0] = activePokemon.status.value/6 if activePokemon.status else 0

        activePokemonStats[0] = activePokemon.stats['hp'] /500 if 'hp' in  activePokemon.stats and activePokemon.stats['hp'] else -1
        activePokemonStats[1] = activePokemon.stats['atk']/500 if 'atk' in activePokemon.stats and activePokemon.stats['atk'] else -1
        activePokemonStats[2] = activePokemon.stats['def']/500 if 'def' in activePokemon.stats and activePokemon.stats['def'] else -1
        activePokemonStats[3] = activePokemon.stats['spa']/500 if 'spa' in activePokemon.stats and activePokemon.stats['spa'] else -1
        activePokemonStats[4] = activePokemon.stats['spd']/500 if 'spd' in activePokemon.stats and activePokemon.stats['spd'] else -1
        activePokemonStats[5] = activePokemon.stats['spe']/500 if 'spe' in activePokemon.stats and activePokemon.stats['spe'] else -1

        for field,turn in battle.fields.items():
            activeFields[field.value - 1] = 1

        for sideCondition,val in battle.opponent_side_conditions.items():
            opponentSideConditions[sideCondition.value - 1] = 1
        
        for sideCondition,val in battle.side_conditions.items():
            activePokemonSideConditions[sideCondition.value - 1] = 1

        for i,pokemon in enumerate(battle.available_switches):
            firstTypeMultiplyer = pokemon.type_1.damage_multiplier(
                    opponentActivePokemon.type_1,
                    opponentActivePokemon.type_2,)
            teamDmgMultiplyer[i] = firstTypeMultiplyer

            if pokemon.type_2 != None:
                secondTypeMultiplyer = pokemon.type_2.damage_multiplier(
                        opponentActivePokemon.type_1,
                        opponentActivePokemon.type_2,)
                teamDmgMultiplyer[i] *= secondTypeMultiplyer
            teamDmgMultiplyer[i] /= 4

        for i,pokemon in enumerate(battle.team.values()):
            i = i*2
            if pokemon.fainted:
                teamTypes[i] = 0
                teamTypes[i + 1] = 0
            else:
                teamTypes[i] = pokemon.type_1.value/19 if pokemon.type_1 != None else 0
                teamTypes[i + 1] =  pokemon.type_2.value/19 if pokemon.type_2 != None else 0
                teamHealth[i//2] = pokemon.current_hp/800 if pokemon.current_hp else 0 #divide by maximum possible HP


        for i, pokemon in enumerate(battle.opponent_team.values()):
            i = i*2
            if pokemon.fainted:
                opponentTeamTypes[i] = -1
                opponentTeamTypes[i + 1] = -1
            else:
                opponentTeamTypes[i] = pokemon.type_1.value/19 if pokemon.type_1 != None else 0
                opponentTeamTypes[i + 1] =  pokemon.type_2.value/19 if pokemon.type_2 != None else 0
                opponentTeamHealth[i//2] = pokemon.current_hp/800 if pokemon.current_hp else 0 #divide by maximum possible HP

        for i, move in enumerate(battle.available_moves):
            activePokemonMovesStatusEffects[i] = move.status.value/7 if move.status != None else 0
            activePokemonMovesBasePower[i] = (
                move.base_power / 300
            )
            # Simple rescaling to facilitate learning
            if move.type and activePokemonMovesBasePower[i] > 0:
                activePokemonMovesDmgMultiplier[i] = move.type.damage_multiplier(
                    opponentActivePokemon.type_1,
                    opponentActivePokemon.type_2,) / 4

        # We count how many pokemons have fainted in each team
        faintedTeamPokemon = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        faintedOpponentTeamPokemon = (len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6)
        #put -1 for inputs if the pokemon is already fainted
        #for moves, actually show the status effect that is used not just a 1

        # print()
        # print()
        # print("activePokemonMovesBasePower: ",activePokemonMovesBasePower)
        # print("activePokemonMovesDmgMultiplier:",activePokemonMovesDmgMultiplier)
        # print("canDynamax:",canDynamax)
        # print("teamTypes:",teamTypes)
        # print("opponentTeamTypes:",opponentTeamTypes)
        # print("teamDmgMultiplyer :",teamDmgMultiplyer)
        # print("dynamaxTurn:",dynamaxTurn)
        # print("activePokemonMovesStatusEffects :",activePokemonMovesStatusEffects)
        # print("currentWeather :",currentWeather)
        # print("opponentDyanamaxTurn:",opponentDyanamaxTurn)
        # print("opponentCanDynamax:",opponentCanDynamax)
        # print("opponentSideConditions:",opponentSideConditions)
        # print("teamHealth :",teamHealth)
        # print("opponentTeamHealth :",opponentTeamHealth)
        # print("activeFields:",activeFields)
        # print("activePokemonSideConditions:",activePokemonSideConditions)
        # print("activeOpponentPokemonStatus :",activeOpponentPokemonStatus)
        # print("activePokemonStatus :",activePokemonStatus)
        # print("activePokemonStats :",activePokemonStats)
        # print("faintedTeamPokemon: ",faintedTeamPokemon)
        # print("faintedOpponentTeamPokemon: ",faintedOpponentTeamPokemon)
        # print()
        # print()

        # Final vector with 12 components
        final_vector = np.concatenate(
            [
                activePokemonMovesBasePower,
                activePokemonMovesDmgMultiplier,
                [faintedTeamPokemon, faintedOpponentTeamPokemon],
                canDynamax,
                dynamaxTurn,
                teamTypes,
                opponentTeamTypes,
                teamDmgMultiplyer,
                activePokemonMovesStatusEffects,
                currentWeather,
                opponentDyanamaxTurn,
                opponentCanDynamax,
                opponentSideConditions,
                teamHealth,
                opponentTeamHealth,
                activeFields,
                activePokemonSideConditions,
                activeOpponentPokemonStatus,
                activePokemonStatus,
                activePokemonStats
            ]
        )
        return np.float32(final_vector)

    def describe_embedding(self) -> Space:
        low = []
        moveBasePowerLower = [-1]*4
        moveDamageMultiplyerLower = [-1]*4
        faintedTeamLower = [0]
        faintedOpponentTeamLower = [0]
        canDynamaxLower = [0]
        dynamaxTurnLower = [-1]
        teamTypeLower= [0]*12
        opponentTeamTypeLower = [0]*12
        teamMultiplyerLower = [-1]*6
        moveStatusLower = [-1]*4
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
        activePokemonStatsLower = [-1]*6

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
        low += activePokemonStatsLower
        

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
        activePokemonStatsUpper = [1]*6
        
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
        high += activePokemonStatsUpper

        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )
    
def buildModelLayers(model,inputShape, outputLen):
    model.add(Dense(inputShape[1], activation="elu", input_shape=inputShape))
    model.add(Normalization())
    model.add(Flatten())
    model.add(Dense((inputShape[1] + outputLen)//2, activation="elu"))
    model.add(Normalization())
    model.add(Dense(outputLen, activation="linear"))

def trainAgainstAgent(dqn, steps, trainingEnv, agent, restart = False):
    if restart : trainingEnv.reset_env(restart=True, opponent=agent)
    dqn.fit(trainingEnv, nb_steps=steps)

def evalAgainstAgent(dqn,evalEnv,agent, agentName, restart = False):
    print()
    # Evaluating the model
    if restart : evalEnv.reset_env(restart=True, opponent=agent)
    print("Results against" + agentName + "player:")
    dqn.test(evalEnv, nb_episodes=50, verbose=False, visualize=False)
    print(
        f"DQN Evaluation: {evalEnv.n_won_battles} victories out of {evalEnv.n_finished_battles} episodes"
    )
    print()

def evalWithUtilMethod(dqn, evalEnv):
    print()
    print("Evalutation with Util Method starting ------------------------------------")
    evalEnv.reset_env(restart=False)
    # Evaluate the player with included util method
    n_challenges = 250
    placement_battles = 40
    eval_task = background_evaluate_player(
        evalEnv.agent, n_challenges, placement_battles
    )
    dqn.test(evalEnv, nb_episodes=n_challenges, verbose=False, visualize=False)
    print("Evaluation with included method:", eval_task.result())
    print()

def crossEval(dqn, evalEnv,currentFile):
    print()
    currentFile.write("Cross Evaluating against all agents starting ------------------------------------")
    # Cross evaluate the player with included util method
    evalEnv.reset_env(restart = False)
    n_challenges = 50
    players = [
        evalEnv.agent,
        RandomPlayer(battle_format="gen8randombattle"),
        MaxBasePowerPlayer(battle_format="gen8randombattle"),
        SimpleHeuristicsPlayer(battle_format="gen8randombattle"),
    ]
    cross_eval_task = background_cross_evaluate(players, n_challenges)
    dqn.test(
        evalEnv,
        nb_episodes=n_challenges * (len(players) - 1),
        verbose=False,
        visualize=False,
    )
    cross_evaluation = cross_eval_task.result()
    table = [["-"] + [p.username for p in players]]
    for p_1, results in cross_evaluation.items():
        table.append([p_1] + [cross_evaluation[p_1][p_2] for p_2 in results])
    currentFile.write("Cross evaluation of DQN with baselines:")
    currentFile.write(tabulate(table))
    print()

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
    
def evalAllDqns(evalEnv,dqnDict, currentFile):
    for trainingTimes,dqn in dqnDict.items():
        print()
        currentFile.write(f'Random Training Steps: " {trainingTimes[0]} " Max Damage Agent Training Steps: " {trainingTimes[1]} " Heuristics Agent Training Steps: " {trainingTimes[2]} " ------')
        crossEval(dqn, evalEnv, currentFile)
        print()

def checkCurrentEnvironment():
    # First test the environment to ensure the class is consistent
    # with the OpenAI API
    #10 tries
    for i in range(10):
        try:
            randomAgent = RandomPlayer(battle_format="gen8randombattle")
            test_env = SimpleRLPlayer(battle_format="gen8randombattle", start_challenging=True, opponent=randomAgent)
            check_env(test_env)
            test_env.close()
            print("Test Environment Closed")
            return
        except:
            continue
    


async def main():
    checkCurrentEnvironment()
    # Create one environment for training and one for evaluation
    trainEnv = SimpleRLPlayer(battle_format="gen8randombattle", opponent=RandomPlayer(battle_format="gen8randombattle"), start_challenging=True)
    trainEnv = wrap_for_old_gym_api(trainEnv)

    evalEnv = SimpleRLPlayer(
        battle_format="gen8randombattle", opponent=RandomPlayer(battle_format="gen8randombattle"), start_challenging=True
    )
    evalEnv = wrap_for_old_gym_api(evalEnv)

    # Compute dimensions
    n_action = trainEnv.action_space.n
    input_shape = (1,) + trainEnv.observation_space.shape
    print(input_shape)

    # Create model
    model = Sequential()
    buildModelLayers(model, input_shape, n_action)

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

    dqnDict = {}
    randomAgent = RandomPlayer(battle_format="gen8randombattle")
    maxAgent = MaxBasePowerPlayer(battle_format="gen8randombattle")
    heuristicsAgent = SimpleHeuristicsPlayer(battle_format="gen8randombattle")

    trainingTuner(model,n_action,policy,memory,trainEnv,dqnDict, randomAgent,maxAgent,heuristicsAgent, 3)
    trainEnv.close()

    print("Attempting to run Evals and save to file --------")
    try:
        currentFile = open("evalutationResults.txt", "w")
        evalAllDqns(evalEnv,dqnDict,currentFile)
    finally:
        currentFile.close()
    
        
    evalEnv.close()

    # dqn.save_weights("Saved Models/currentModel")


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
