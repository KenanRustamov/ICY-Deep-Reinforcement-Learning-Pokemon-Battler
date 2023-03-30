import asyncio
import numpy as np

from gym.spaces import Space, Box
from gym.utils.env_checker import check_env
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from tabulate import tabulate
from tensorflow.keras import Input,regularizers
from tensorflow.keras.layers import Dense, Flatten, Normalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import os
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
            current_battle,fainted_value=3.0, hp_value=1.0, victory_value=30.0, status_value= .1
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

        player2dVector = np.full((21, 20), -2)
        opponent2dVector = np.full((21, 20), -2)
        external2dVector = np.full((21, 20), -2)

        threeDimensionalVector = np.full((3, 21, 20), -2)

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
                teamTypes[i] = -1
                teamTypes[i + 1] = -1
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

        # Fill the first row with activePokemonMovesBasePower
        player2dVector[0, :len(activePokemonMovesBasePower)] = activePokemonMovesBasePower

        # Fill the second row with activePokemonMovesDmgMultiplier
        player2dVector[1, :len(activePokemonMovesDmgMultiplier)] = activePokemonMovesDmgMultiplier

        # Fill the third row with faintedTeamPokemon
        player2dVector[2, :1] = faintedTeamPokemon

         # Fill the fifth row with canDynamax
        player2dVector[3, :len(canDynamax)] = canDynamax

        # Fill the sixth row with dynamaxTurn
        player2dVector[4, :len(dynamaxTurn)] = dynamaxTurn

        # Fill the seventh row with teamTypes
        player2dVector[5, :len(teamTypes)] = teamTypes

        # Fill the ninth row with teamDmgMultiplyer
        player2dVector[6, :len(teamDmgMultiplyer)] = teamDmgMultiplyer

        # Fill the tenth row with activePokemonMovesStatusEffects
        player2dVector[7, :len(activePokemonMovesStatusEffects)] = activePokemonMovesStatusEffects

        # Fill the fifteenth row with teamHealth
        player2dVector[8, :len(teamHealth)] = teamHealth

        # Fill the eighteenth row with activePokemonSideConditions
        player2dVector[9, :len(activePokemonSideConditions)] = activePokemonSideConditions

        player2dVector[10, :len(activePokemonStatus)] = activePokemonStatus

        # Fill the twentieth row with activePokemonStats
        player2dVector[11, :len(activePokemonStats)] = activePokemonStats

        # Fill the fourth row with faintedOpponentTeamPokemon
        opponent2dVector[0, :1] = faintedOpponentTeamPokemon
       
        # Fill the eighth row with opponentTeamTypes
        opponent2dVector[1, :len(opponentTeamTypes)] = opponentTeamTypes

        # Fill the twelfth row with opponentDyanamaxTurn
        opponent2dVector[2, :len(opponentDyanamaxTurn)] = opponentDyanamaxTurn

        # Fill the thirteenth row with opponentCanDynamax
        opponent2dVector[3, :len(opponentCanDynamax)] = opponentCanDynamax

        # Fill the fourteenth row with opponentSideConditions
        opponent2dVector[4, :len(opponentSideConditions)] = opponentSideConditions

        # Fill the sixteenth row with opponentTeamHealth
        opponent2dVector[5, :len(opponentTeamHealth)] = opponentTeamHealth

        
        # Fill the nineteenth row with activeOpponentPokemonStatus and activePokemonStatus
        opponent2dVector[6, :len(activeOpponentPokemonStatus)] = activeOpponentPokemonStatus

        # Fill the seventeenth row with activeFields
        external2dVector[0, :len(activeFields)] = activeFields

        # Fill the eleventh row with currentWeather
        external2dVector[1, :len(currentWeather)] = currentWeather

        threeDimensionalVector[0] = player2dVector
        threeDimensionalVector[1] = opponent2dVector
        threeDimensionalVector[2] = external2dVector

        return np.float32(threeDimensionalVector)

    def describe_embedding(self) -> Space:
        low = []
        newLow = np.full((3, 21, 20), -2)
        playerNewLow = np.full((21, 20), -2)
        opponentNewLow = np.full((21, 20), -2)
        externalNewLow = np.full((21, 20), -2)
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

        playerNewLow[0, :len(moveBasePowerLower)] = moveBasePowerLower
        playerNewLow[1, :len(moveDamageMultiplyerLower)] = moveDamageMultiplyerLower
        playerNewLow[2, :len(faintedTeamLower)] = faintedTeamLower
        playerNewLow[3, :len(canDynamaxLower)] = canDynamaxLower
        playerNewLow[4, :len(dynamaxTurnLower)] = dynamaxTurnLower
        playerNewLow[5, :len(teamTypeLower)] = teamTypeLower
        playerNewLow[6, :len(teamMultiplyerLower)] = teamMultiplyerLower
        playerNewLow[7, :len(moveStatusLower)] = moveStatusLower
        playerNewLow[8, :len(teamHealthLower)] = teamHealthLower
        playerNewLow[9, :len(activePokemonSideConditionsLower)] = activePokemonSideConditionsLower
        playerNewLow[10, :len(activePokemonStatusLower)] = activePokemonStatusLower
        playerNewLow[11, :len(activePokemonStatsLower)] = activePokemonStatsLower

        opponentNewLow[0, :len(faintedOpponentTeamLower)] = faintedOpponentTeamLower
        opponentNewLow[1, :len(opponentTeamTypeLower)] = opponentTeamTypeLower
        opponentNewLow[2, :len(opponentDynamaxTurnLower)] = opponentDynamaxTurnLower
        opponentNewLow[3, :len(opponentCanDynamaxLower)] = opponentCanDynamaxLower
        opponentNewLow[4, :len(opponentSideConditionsLower)] = opponentSideConditionsLower
        opponentNewLow[5, :len(opponentTeamHealthLower)] = opponentTeamHealthLower
        opponentNewLow[6, :len(activeOpponentStatusLower)] = activeOpponentStatusLower

        externalNewLow[0, :len(activeFieldsLower)] = activeFieldsLower
        externalNewLow[1, :len(currentWeatherLower)] = currentWeatherLower



        high = []
        newHigh = np.full((3, 21, 20), -2)
        playerNewHigh = np.full((21, 20), -2)
        opponentNewHigh = np.full((21, 20), -2)
        externalNewHigh = np.full((21, 20), -2)
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

        playerNewHigh[0, :len(moveBasePowerUpper)] = moveBasePowerUpper
        playerNewHigh[1, :len(moveDamageMultiplyerUpper)] = moveDamageMultiplyerUpper
        playerNewHigh[2, :len(faintedTeamUpper)] = faintedTeamUpper
        playerNewHigh[3, :len(canDynamaxUpper)] = canDynamaxUpper
        playerNewHigh[4, :len(dynamaxTurnUpper)] = dynamaxTurnUpper
        playerNewHigh[5, :len(teamTypeUpper)] = teamTypeUpper
        playerNewHigh[6, :len(teamMultiplyerUpper)] = teamMultiplyerUpper
        playerNewHigh[7, :len(moveStatusUpper)] = moveStatusUpper
        playerNewHigh[8, :len(teamHealthUpper)] = teamHealthUpper
        playerNewHigh[9, :len(activePokemonSideConditionsUpper)] = activePokemonSideConditionsUpper
        playerNewHigh[10, :len(activePokemonStatusUpper)] = activePokemonStatusUpper
        playerNewHigh[11, :len(activePokemonStatsUpper)] = activePokemonStatsUpper

        opponentNewHigh[0, :len(faintedOpponentTeamUpper)] = faintedOpponentTeamUpper
        opponentNewHigh[1, :len(opponentTeamTypeUpper)] = opponentTeamTypeUpper
        opponentNewHigh[2, :len(opponentDynamaxTurnUpper)] = opponentDynamaxTurnUpper
        opponentNewHigh[3, :len(opponentCanDynamaxUpper)] = opponentCanDynamaxUpper
        opponentNewHigh[4, :len(opponentSideConditionsUpper)] = opponentSideConditionsUpper
        opponentNewHigh[5, :len(opponentTeamHealthUpper)] = opponentTeamHealthUpper
        opponentNewHigh[6, :len(activeOpponentStatusUpper)] = activeOpponentStatusUpper

        externalNewHigh[0, :len(activeFieldsUpper)] = activeFieldsUpper
        externalNewHigh[1, :len(currentWeatherUpper)] = currentWeatherUpper


        newLow[0] = playerNewLow
        newLow[1] = opponentNewLow
        newLow[2] = externalNewLow

        newHigh[0] = playerNewHigh
        newHigh[1] = opponentNewHigh
        newHigh[2] = externalNewHigh

        return Box(
            np.array(newLow, dtype=np.float32),
            np.array(newHigh, dtype=np.float32),
            dtype=np.float32,
        )
    
def buildModelLayers(model,inputShape, outputLen):
    model.add(Dense(inputShape[1], activation="swish", input_shape=inputShape))
    model.add(Normalization())
    model.add(Flatten())
    model.add(Dense((inputShape[1] + outputLen)*2, activation="swish"))
    model.add(Normalization())
    model.add(Dense((inputShape[1] + outputLen)//2, activation="swish"))
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

def getNextNumber():
    newFile = open("currentNumber.txt", "r")
    currentString = newFile.read()
    newFile.close()

    nextNumber = str(int(currentString) + 1)
    newFile = open("currentNumber.txt","w")
    newFile.write(nextNumber)
    newFile.close()
    return nextNumber
    


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

    # trainingTuner(model,n_action,policy,memory,trainEnv,dqnDict, randomAgent,maxAgent,heuristicsAgent, 3)
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

    trainAgainstAgent(dqn, 30000, trainEnv, randomAgent)
    trainAgainstAgent(dqn, 30000, trainEnv, maxAgent, True)
    # trainAgainstAgent(dqn, 30000, trainEnv, heuristicsAgent, True)
    trainEnv.close()
    dqnDict[(30000,30000,30000)] = dqn

    print("Attempting to run Evals and save to file --------")
    nextNum = getNextNumber() 

    modelDir = "C:/Users/kenan/Documents/projects/New Icy/Saved Models/model"+ nextNum
    os.mkdir(modelDir) 
    print("Directory '% s' is built!" % modelDir) 

    try:
        currentFile = open("Saved Models/model" + nextNum + "/evalutationResults.txt", "w")
        evalAllDqns(evalEnv,dqnDict,currentFile)
    finally:
        currentFile.close()
    
        
    evalEnv.close()

    dqn.save_weights("Saved Models/model" + nextNum + "/savedModel")


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
