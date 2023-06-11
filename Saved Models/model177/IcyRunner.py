import asyncio
import numpy as np

from gym.spaces import Space, Box
from gym.utils.env_checker import check_env
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from tabulate import tabulate
from tensorflow.keras import Input,regularizers
from tensorflow.keras.layers import Dense, Flatten, Normalization, Dropout, InputLayer, Conv1D, LSTM, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L1L2
import os
from poke_env import PlayerConfiguration, ShowdownServerConfiguration
from EasySimpleHeuristicPlayer import EasySimpleHeuristicPlayer
from SimpleRLPlayer import SimpleRLPlayer
import sys
import shutil
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
    
def buildModelLayers(model,inputShape, outputLen):
    model.add(Dense(128, activation="swish", input_shape = inputShape))
    model.add(Normalization()) 
    model.add(Flatten())
    model.add(Dropout(.5))
    model.add(Dense(64 , activation="swish"))
    model.add(Normalization()) 
    model.add(Dropout(.3))
    model.add(Dense(22 , activation="swish"))
    model.add(Normalization())
    model.add(Dropout(.1))
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
    print("Cross Evaluating against all agents starting ------------------------------------")
    currentFile.write("Cross Evaluating against all agents starting ------------------------------------")
    # Cross evaluate the player with included util method
    evalEnv.reset_env(restart = False)
    n_challenges = 100
    players = [
        evalEnv.agent,
        RandomPlayer(battle_format="gen8randombattle"),
        MaxBasePowerPlayer(battle_format="gen8randombattle"),
        EasySimpleHeuristicPlayer(battle_format="gen8randombattle"),
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
    
def evalAllDqns(evalEnv,dqnDict, currentFile):
    for trainingTimes,dqn in dqnDict.items():
        print()
        sys.stdout = currentFile
        # dqn.mos
        sys.stdout = sys.__stdout__
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

def connectToRemote(username, password):
    player = RandomPlayer(
        player_configuration=PlayerConfiguration(username, password),
        server_configuration=ShowdownServerConfiguration,
    )
    
#Pass none for challengerName to make it accept challenges from anyone
async def acceptChallenges(botPlayer, challengerName):
    await botPlayer.accept_challenges(challengerName,1)

async def sendChallenges(botPlayer, challengerName, numChallenges):
    await botPlayer.send_challenges(challengerName, n_challenges=numChallenges)

async def playOnLadder(botPlayer,numOfGames):
    await botPlayer.ladder(numOfGames)
    for battle in botPlayer.battles.values():
        print(battle.rating, battle.opponent_rating)

def loadWeights(model, savedWeightsPath):
    print("Loading Weights from path: " + savedWeightsPath)
    model.load_weights(savedWeightsPath)

def createAndReturnDqnAgent(n_action,input_shape):
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
    dqn.compile(Adam(learning_rate=0.00025, amsgrad=True), metrics=["mae"])
    model.summary()
    return dqn

def createEnvironment(agent):
    trainEnv = SimpleRLPlayer(battle_format="gen8randombattle", opponent=agent, start_challenging=True)
    trainEnv = wrap_for_old_gym_api(trainEnv)
    return trainEnv

async def main():
    # checkCurrentEnvironment()
    # Create one environment for training and one for evaluation
    trainEnv = createEnvironment(RandomPlayer(battle_format="gen8randombattle"))
    evalEnv = createEnvironment(RandomPlayer(battle_format="gen8randombattle"))

    # Compute dimensions
    n_action = trainEnv.action_space.n
    input_shape = (1,) + trainEnv.observation_space.shape

    randomAgent = RandomPlayer(battle_format="gen8randombattle")
    maxAgent = MaxBasePowerPlayer(battle_format="gen8randombattle")
    heuristicsAgent = SimpleHeuristicsPlayer(battle_format="gen8randombattle")
    easyHeuristicAgent = EasySimpleHeuristicPlayer(battle_format="gen8randombattle")

    # trainingTuner(model,n_action,policy,memory,trainEnv,dqnDict, randomAgent,maxAgent,heuristicsAgent, 3)
    dqn = createAndReturnDqnAgent(n_action,input_shape)
    # loadWeights(dqn, "Saved Models/model133/savedModel")
    
    randomStep = 100000
    maxStep = 0
    easyHeuristicStep = 0
    heuristicStep = 0

    trainAgainstAgent(dqn, randomStep, trainEnv, randomAgent)
    # trainAgainstAgent(dqn, maxStep, trainEnv, maxAgent,True)
    # trainAgainstAgent(dqn, easyHeuristicStep, trainEnv, easyHeuristicAgent,True)
    # trainAgainstAgent(dqn, heuristicStep, trainEnv, heuristicsAgent, True)
    trainEnv.close()
    dqnDict = {}
    dqnDict[(randomStep,maxStep,easyHeuristicStep)] = dqn

    print("Attempting to run Evals and save to file --------")

    nextNum = getNextNumber() 
    modelDir = "C:/Users/kenan/Documents/projects/New Icy/Saved Models/model"+ nextNum
    os.mkdir(modelDir) 

    print("Directory '% s' is built!" % modelDir) 
    relativePath = "Saved Models/model" + nextNum
    dqn.save_weights(relativePath + "/savedModel")
    try:
        currentFile = open(relativePath + "/evalutationResults.txt", "w")
        evalAllDqns(evalEnv,dqnDict,currentFile)
    finally:
        currentFile.close()

    shutil.copyfile("./IcyRunner.py", relativePath + "/IcyRunner.py")
    shutil.copyfile("./SimpleRLPlayer.py", relativePath + "/SimpleRLPlayer.py")

    evalEnv.close()
    


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
