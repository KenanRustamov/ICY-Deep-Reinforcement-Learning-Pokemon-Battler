import numpy as np

from gym.spaces import Space, Box
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
import hashlib


def wordToNumber(word):
    hash_object = hashlib.sha1(word.encode())
    hex_dig = hash_object.hexdigest()
    int_val = int(hex_dig, 16)
    max_val = 2**160 - 1 # maximum value of SHA-1 hash
    return float(int_val % 2**32) / 2**32

class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle,fainted_value=1, victory_value=30.0, hp_value=.1, status_value=.5
        )
    
    def calculateTypeAdvantage(self,types, opponent):
        typeAdvantage = 1
        typeAdvantage = opponent.damage_multiplier(types[0]) if types[0] is not None else 1 * opponent.damage_multiplier(types[1]) if types[1] is not None else 1
        return typeAdvantage

    def embed_battle(self, battle: AbstractBattle) -> ObservationType:
        # -1 indicates that the move does not have a base power
        # or is not available
        activePokemon = battle.active_pokemon
        opponentActivePokemon = battle.opponent_active_pokemon

        activePokemonMovesBasePower = -np.ones(4)
        activePokemonMovesDmgMultiplier = -np.ones(4)
        faintedTeamPokemon = [len([mon for mon in battle.team.values() if mon.fainted]) / 6]
        canDynamax = np.ones(1)
        dynamaxTurn = np.ones(1)
        teamTypes = np.zeros(12)
        teamDmgMultiplyer = -np.ones(6)
        activePokemonMovesStatusEffects = -np.ones(4)
        teamHealth = np.zeros(6)
        activePokemonSideConditions = np.zeros(20)
        activePokemonStatus = np.zeros(1)
        activePokemonStats = -np.ones(5)
        activePokemonAbility = np.zeros(1)
        activePokemonStatusCounter = np.zeros(1)
        activePokemonMovesAccuracy = -np.ones(4)
        activePokemonMovesCritRatio = -np.ones(4)
        activePokemonMovesCurrentPp = -np.ones(4)
        activePokemonMovesExpectedHits = -np.ones(4)
        activePokemonMovesForceSwitch = -np.ones(4)
        activePokemonMovesHeals = -np.ones(4)
        activePokemonMovesPriority = -np.ones(4)
        activePokemonMovesRecoil = -np.ones(4)
        activePokemonMovesSideConditions = -np.ones(4)
        activePokemonMovesTerrain = -np.ones(4)
        activePokemonMovesWeather = -np.ones(4)
        faintedOpponentTeamPokemon = [len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6]
        opponentTeamTypes = np.zeros(12)
        opponentDyanamaxTurn = np.ones(1)
        opponentCanDynamax = np.ones(1)
        opponentSideConditions = np.zeros(20)
        opponentTeamHealth = np.zeros(6)
        activeOpponentPokemonStatus = np.zeros(1)
        activeOpponentPokemonAbility = np.zeros(1)
        activeFields = np.zeros(12)
        currentWeather = np.zeros(1)
        pokemonIds = -np.ones(6)
        opponentPokemonIds = -np.ones(6)
        activePokemonMovesIds = -np.ones(4)
        opponentActivePokemonIds = -np.ones(4)
        pokemonMatchupAgainstOpponent = -np.ones(6)
        opponentPokemonMatchupAgainstPokemon = -np.ones(6)
        pokemonStatuses = -np.ones(6)

        canDynamax[0] = 1 if battle.can_dynamax else 0
        dynamaxTurn[0] = battle.dynamax_turns_left/3 if battle.dynamax_turns_left != None else -1
        opponentDyanamaxTurn[0] = battle.opponent_dynamax_turns_left/3 if battle.opponent_dynamax_turns_left != None else -1
        currentWeather[0] = -1 if not battle.weather else list(battle.weather.items())[0][0].value/8
        opponentCanDynamax[0] = 1 if battle.opponent_can_dynamax else 0
        activeOpponentPokemonStatus[0] = opponentActivePokemon.status.value/7 if opponentActivePokemon.status else 0
        activePokemonStatus[0] = activePokemon.status.value/7 if activePokemon.status else 0
        # activePokemonAbility[0] = -1 if not activePokemon.ability else wordToNumber(activePokemon.ability)
        # activeOpponentPokemonAbility[0] = -1 if not opponentActivePokemon.ability else wordToNumber(opponentActivePokemon.ability)
        # activePokemonStatusCounter[0] = -1 if not activePokemon.status_counter else activePokemon.status_counter/100

        activePokemonStats[0] = activePokemon.stats['atk'] if 'atk' in activePokemon.stats and activePokemon.stats['atk'] else -1
        activePokemonStats[1] = activePokemon.stats['def'] if 'def' in activePokemon.stats and activePokemon.stats['def'] else -1
        activePokemonStats[2] = activePokemon.stats['spa'] if 'spa' in activePokemon.stats and activePokemon.stats['spa'] else -1
        activePokemonStats[3] = activePokemon.stats['spd'] if 'spd' in activePokemon.stats and activePokemon.stats['spd'] else -1
        activePokemonStats[4] = activePokemon.stats['spe'] if 'spe' in activePokemon.stats and activePokemon.stats['spe'] else -1

        # print(activePokemon, activePokemonStats)

        # for field,turn in battle.fields.items():
        #     activeFields[field.value - 1] = 1

        # for sideCondition,val in battle.opponent_side_conditions.items():
        #     opponentSideConditions[sideCondition.value - 1] = 1
        
        # for sideCondition,val in battle.side_conditions.items():
        #     activePokemonSideConditions[sideCondition.value - 1] = 1

        # for i,pokemon in enumerate(battle.available_switches):
        #     firstTypeMultiplyer = pokemon.type_1.damage_multiplier(
        #             opponentActivePokemon.type_1,
        #             opponentActivePokemon.type_2,)
        #     teamDmgMultiplyer[i] = firstTypeMultiplyer

        #     if pokemon.type_2 != None:
        #         secondTypeMultiplyer = pokemon.type_2.damage_multiplier(
        #                 opponentActivePokemon.type_1,
        #                 opponentActivePokemon.type_2,)
        #         teamDmgMultiplyer[i] *= secondTypeMultiplyer
        #     teamDmgMultiplyer[i] /= 4

        for i,pokemon in enumerate(battle.team.values()):
            i = i
            if pokemon.fainted:
                teamTypes[i*2] = -1
                teamTypes[i*2 + 1] = -1
                pokemonIds[i] = -1
                pokemonMatchupAgainstOpponent[i] = -1
            else:
                teamTypes[i*2] = pokemon.type_1.value/19 if pokemon.type_1 != None else 0
                teamTypes[i*2 + 1] =  pokemon.type_2.value/19 if pokemon.type_2 != None else 0
                teamHealth[i] = pokemon.current_hp/800 if pokemon.current_hp else 0 #divide by maximum possible HP
                pokemonIds[i] = wordToNumber(pokemon.species)
                pokemonMatchupAgainstOpponent[i] = self.calculateTypeAdvantage(pokemon.types, opponentActivePokemon)/4
                pokemonStatuses[i] = pokemon.status.value/7 if pokemon.status else 0

        for i, pokemon in enumerate(battle.opponent_team.values()):
            i = i*2
            if pokemon.fainted:
                opponentTeamTypes[i] = -1
                opponentTeamTypes[i + 1] = -1
                opponentPokemonIds[i//2] = -1
            else:
                opponentTeamTypes[i] = pokemon.type_1.value/19 if pokemon.type_1 != None else 0
                opponentTeamTypes[i + 1] =  pokemon.type_2.value/19 if pokemon.type_2 != None else 0
                opponentTeamHealth[i//2] = pokemon.current_hp/800 if pokemon.current_hp else 0 #divide by maximum possible HP
                opponentPokemonIds[i//2] = wordToNumber(pokemon.species)

        for i, move in enumerate(battle.available_moves):
            activePokemonMovesStatusEffects[i] = move.status.value/7 if move.status != None else 0
            activePokemonMovesBasePower[i] = (
                move.base_power / 300
            )
            # activePokemonMovesAccuracy[i] = move.accuracy
            # activePokemonMovesCritRatio[i] = move.crit_ratio

            # activePokemonMovesCurrentPp[i] = -1 if not move.current_pp else move.current_pp
            # activePokemonMovesExpectedHits[i] = -1 if not move.expected_hits else move.expected_hits/5
            # activePokemonMovesForceSwitch[i] = 0 if not move.force_switch else 1
            # activePokemonMovesHeals[i] = -1 if not move.heal else move.heal
            # activePokemonMovesPriority[i] = move.priority
            # activePokemonMovesRecoil[i] = -1 if not move.recoil else move.recoil
            # activePokemonMovesSideConditions[i] = -1 if not move.side_condition else wordToNumber(move.side_condition)
            # activePokemonMovesTerrain[i] = -1 if not move.terrain else move.terrain.value
            # activePokemonMovesWeather[i] = -1 if not move.weather else move.weather.value
            # activePokemonMovesIds[i] = -1 if not move.id else wordToNumber(move.id)

            activePokemonMovesDmgMultiplier[i] = opponentActivePokemon.damage_multiplier(move) if move.base_power else -1

        inputVector = np.concatenate(
            [
                activePokemonMovesBasePower,
                activePokemonMovesDmgMultiplier,
                # faintedTeamPokemon,
                canDynamax,
                dynamaxTurn,
                # teamTypes,
                # teamDmgMultiplyer,
                activePokemonMovesStatusEffects,
                teamHealth,
                # activePokemonSideConditions,
                activePokemonStatus,
                # activePokemonStats,
                # activePokemonAbility,
                # activePokemonStatusCounter,
                # activePokemonMovesAccuracy,
                # activePokemonMovesCritRatio,
                # activePokemonMovesCurrentPp,
                # activePokemonMovesExpectedHits,
                # activePokemonMovesForceSwitch,
                # activePokemonMovesHeals,
                # activePokemonMovesPriority,
                # activePokemonMovesRecoil,
                # activePokemonMovesSideConditions,
                # activePokemonMovesTerrain,
                # activePokemonMovesWeather,
                # faintedOpponentTeamPokemon,
                # opponentTeamTypes,
                # opponentDyanamaxTurn,
                # opponentCanDynamax,
                # opponentSideConditions,
                opponentTeamHealth,
                activeOpponentPokemonStatus,
                # activeOpponentPokemonAbility,
                # activeFields,
                # currentWeather,
                pokemonIds,
                opponentPokemonIds,
                activePokemonMovesIds,
                pokemonMatchupAgainstOpponent,
                pokemonStatuses
            ]
        )

        return np.float32(inputVector)

    def describe_embedding(self) -> Space:
        activePokemonMovesBasePower = [[-1]*4,[1]*4]
        activePokemonMovesDmgMultiplier = [[-1]*4,[1]*4]
        faintedTeamPokemon = [[0],[1]]
        canDynamax = [[0],[1]]
        dynamaxTurn = [[-1],[1]]
        teamTypes = [[0]*12,[1]*12]
        teamDmgMultiplyer = [[-1]*6,[1]*6]
        activePokemonMovesStatusEffects = [[-1]*4,[1]*4]
        teamHealth = [[0]*6,[1]*6]
        activePokemonSideConditions  = [[0]*20,[1]*20]
        activePokemonStatus = [[-1],[1]]
        activePokemonStats = [[-1]*5,[1]*5]
        activePokemonAbility = [[-1]*1,[1]*1]
        activePokemonStatusCounter = [[-1]*1,[1]*1]
        activePokemonMovesAccuracy = [[-1]*4,[1]*4]
        activePokemonMovesCritRatio = [[-1]*4,[1]*4]
        activePokemonMovesCurrentPp = [[-1]*4,[1]*4]
        activePokemonMovesExpectedHits = [[-1]*4,[1]*4]
        activePokemonMovesForceSwitch = [[0]*4,[1]*4]
        activePokemonMovesHeals = [[-1]*4,[1]*4]
        activePokemonMovesPriority = [[-1]*4,[1]*4]
        activePokemonMovesRecoil = [[-1]*4,[1]*4]
        activePokemonMovesSideConditions = [[-1]*4,[1]*4]
        activePokemonMovesTerrain = [[-1]*4,[1]*4]
        activePokemonMovesWeather = [[-1]*4,[1]*4]
        faintedOpponentTeamPokemon = [[0],[1]]
        opponentTeamTypes = [[0]*12,[1]*12]
        opponentDyanamaxTurn = [[-1],[1]]
        opponentCanDynamax = [[0],[1]]
        opponentSideConditions = [[0]*20,[1]*20]
        opponentTeamHealth = [[0]*6,[1]*6]
        activeOpponentPokemonStatus = [[-1],[1]]
        activeOpponentPokemonAbility = [[-1]*1,[1]*1]
        activeFields = [[0]*12,[0]*12]
        currentWeather = [[-1],[1]]
        pokemonIds = [[-1]*6,[1]*6]
        opponentPokemonIds = [[-1]*6,[1]*6]
        activePokemonMovesIds = [[-1]*4,[1]*4]
        pokemonMatchupAgainstOpponent = [[-1]*6,[1]*6]
        pokemonStatuses = [[-1]*6,[1]*6]

        inputVectorLow =np.concatenate( 
        [
            activePokemonMovesBasePower[0],
            activePokemonMovesDmgMultiplier[0],
            # faintedTeamPokemon[0],
            canDynamax[0],
            dynamaxTurn[0],
            # teamTypes[0],
            # teamDmgMultiplyer[0],
            activePokemonMovesStatusEffects[0],
            teamHealth[0],
            # activePokemonSideConditions[0],
            activePokemonStatus[0],
            # activePokemonStats[0],
            # activePokemonAbility[0],
            # activePokemonStatusCounter[0],
            # activePokemonMovesAccuracy[0],
            # activePokemonMovesCritRatio[0],
            # activePokemonMovesCurrentPp[0],
            # activePokemonMovesExpectedHits[0],
            # activePokemonMovesForceSwitch[0],
            # activePokemonMovesHeals[0],
            # activePokemonMovesPriority[0],
            # activePokemonMovesRecoil[0],
            # activePokemonMovesSideConditions[0],
            # activePokemonMovesTerrain[0],
            # activePokemonMovesWeather[0],
            # faintedOpponentTeamPokemon[0],
            # opponentTeamTypes[0],
            # opponentDyanamaxTurn[0],
            # opponentCanDynamax[0],
            # opponentSideConditions[0],
            opponentTeamHealth[0],
            activeOpponentPokemonStatus[0],
            # activeOpponentPokemonAbility[0],
            # activeFields[0],
            # currentWeather[0],
            pokemonIds[0],
            opponentPokemonIds[0],
            activePokemonMovesIds[0],
            pokemonMatchupAgainstOpponent[0],
            pokemonStatuses[0]
        ] )

        inputVectorHigh = np.concatenate (
            [
                activePokemonMovesBasePower[1],
                activePokemonMovesDmgMultiplier[1],
                # faintedTeamPokemon[1],
                canDynamax[1],
                dynamaxTurn[1],
                # teamTypes[1],
                # teamDmgMultiplyer[1],
                activePokemonMovesStatusEffects[1],
                teamHealth[1],
                # activePokemonSideConditions[1],
                activePokemonStatus[1],
                # activePokemonStats[1],
                # activePokemonAbility[1],
                # activePokemonStatusCounter[1],
                # activePokemonMovesAccuracy[1],
                # activePokemonMovesCritRatio[1],
                # activePokemonMovesCurrentPp[1],
                # activePokemonMovesExpectedHits[1],
                # activePokemonMovesForceSwitch[1],
                # activePokemonMovesHeals[1],
                # activePokemonMovesPriority[1],
                # activePokemonMovesRecoil[1],
                # activePokemonMovesSideConditions[1],
                # activePokemonMovesTerrain[1],
                # activePokemonMovesWeather[1],
                # faintedOpponentTeamPokemon[1],
                # opponentTeamTypes[1],
                # opponentDyanamaxTurn[1],
                # opponentCanDynamax[1],
                # opponentSideConditions[1],
                opponentTeamHealth[1],
                activeOpponentPokemonStatus[1],
                # activeOpponentPokemonAbility[1],
                # activeFields[1],
                # currentWeather[1],
                pokemonIds[1],
                opponentPokemonIds[1],
                activePokemonMovesIds[1],
                pokemonMatchupAgainstOpponent[1],
                pokemonStatuses[1]
            ]
            )

        return Box(
            np.array(inputVectorLow, dtype=np.float32),
            np.array(inputVectorHigh, dtype=np.float32),
            dtype=np.float32,
        )