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
            current_battle,fainted_value=3.0, hp_value=1.0, victory_value=30.0, status_value= .1
        )

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
        activePokemonStats = -np.ones(6)
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

        canDynamax[0] = 1 if battle.can_dynamax else 0
        dynamaxTurn[0] = battle.dynamax_turns_left/3 if battle.dynamax_turns_left != None else -1
        opponentDyanamaxTurn[0] = battle.opponent_dynamax_turns_left/3 if battle.opponent_dynamax_turns_left != None else -1
        currentWeather[0] = -1 if not battle.weather else list(battle.weather.items())[0][0].value/8
        opponentCanDynamax[0] = 1 if battle.opponent_can_dynamax else 0
        activeOpponentPokemonStatus[0] = opponentActivePokemon.status.value/6 if opponentActivePokemon.status else 0
        activePokemonStatus[0] = activePokemon.status.value/6 if activePokemon.status else 0
        activePokemonAbility[0] = -1 if not activePokemon.ability else wordToNumber(activePokemon.ability)
        activeOpponentPokemonAbility[0] = -1 if not opponentActivePokemon.ability else wordToNumber(opponentActivePokemon.ability)
        activePokemonStatusCounter[0] = -1 if not activePokemon.status_counter else activePokemon.status_counter/100

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
            activePokemonMovesAccuracy[i] = move.accuracy
            activePokemonMovesCritRatio[i] = move.crit_ratio

            activePokemonMovesCurrentPp[i] = -1 if not move.current_pp else move.current_pp
            activePokemonMovesExpectedHits[i] = -1 if not move.expected_hits else move.expected_hits/5
            activePokemonMovesForceSwitch[i] = 0 if not move.force_switch else 1
            activePokemonMovesHeals[i] = -1 if not move.heal else move.heal
            activePokemonMovesPriority[i] = move.priority
            activePokemonMovesRecoil[i] = -1 if not move.recoil else move.recoil
            activePokemonMovesSideConditions[i] = -1 if not move.side_condition else wordToNumber(move.side_condition)
            activePokemonMovesTerrain[i] = -1 if not move.terrain else move.terrain.value
            activePokemonMovesWeather[i] = -1 if not move.weather else move.weather.value

            if move.type and activePokemonMovesBasePower[i] > 0:
                activePokemonMovesDmgMultiplier[i] = move.type.damage_multiplier(
                    opponentActivePokemon.type_1,
                    opponentActivePokemon.type_2,) / 4

        inputVector = np.concatenate(
            [
                activePokemonMovesBasePower,
                activePokemonMovesDmgMultiplier,
                faintedTeamPokemon,
                canDynamax,
                dynamaxTurn,
                teamTypes,
                teamDmgMultiplyer,
                activePokemonMovesStatusEffects,
                teamHealth,
                activePokemonSideConditions,
                activePokemonStatus,
                activePokemonStats,
                activePokemonAbility,
                activePokemonStatusCounter,
                activePokemonMovesAccuracy,
                activePokemonMovesCritRatio,
                activePokemonMovesCurrentPp,
                activePokemonMovesExpectedHits,
                activePokemonMovesForceSwitch,
                activePokemonMovesHeals,
                activePokemonMovesPriority,
                activePokemonMovesRecoil,
                activePokemonMovesSideConditions,
                activePokemonMovesTerrain,
                activePokemonMovesWeather,
                faintedOpponentTeamPokemon,
                opponentTeamTypes,
                opponentDyanamaxTurn,
                opponentCanDynamax,
                opponentSideConditions,
                opponentTeamHealth,
                activeOpponentPokemonStatus,
                activeOpponentPokemonAbility,
                activeFields,
                currentWeather
            ]
        )

        return np.float32(inputVector)

    def describe_embedding(self) -> Space:
        activePokemonMovesBasePowerLower = [-1]*4
        activePokemonMovesDmgMultiplierLower = [-1]*4
        faintedTeamPokemonLower = [0]
        canDynamaxLower = [0]
        dynamaxTurnLower = [-1]
        teamTypesLower = [0]*12
        teamDmgMultiplyerLower = [-1]*6
        activePokemonMovesStatusEffectsLower = [-1]*4
        teamHealthLower = [0]*6
        activePokemonSideConditionsLower  = [0]*20
        activePokemonStatusLower = [0]
        activePokemonStatsLower = [-1]*6
        activePokemonAbilityLower = [-1]*1
        activePokemonStatusCounterLower = [-1]*1
        activePokemonMovesAccuracyLower = [-1]*4
        activePokemonMovesCritRatioLower = [-1]*4
        activePokemonMovesCurrentPpLower = [-1]*4
        activePokemonMovesExpectedHitsLower = [-1]*4
        activePokemonMovesForceSwitchLower = [0]*4
        activePokemonMovesHealsLower = [-1]*4
        activePokemonMovesPriorityLower = [-1]*4
        activePokemonMovesRecoilLower = [-1]*4
        activePokemonMovesSideConditionsLower = [-1]*4
        activePokemonMovesTerrainLower = [-1]*4
        activePokemonMovesWeatherLower = [-1]*4
        faintedOpponentTeamPokemonLower = [0]
        opponentTeamTypesLower = [0]*12
        opponentDyanamaxTurnLower = [-1]
        opponentCanDynamaxLower = [0]
        opponentSideConditionsLower = [0]*20
        opponentTeamHealthLower = [0]*6
        activeOpponentPokemonStatusLower = [0]
        activeOpponentPokemonAbilityLower = [-1]*1
        activeFieldsLower = [0]*12
        currentWeatherLower = [-1]


        activePokemonMovesBasePowerUpper = [1]*4
        activePokemonMovesDmgMultiplierUpper = [1]*4
        faintedTeamPokemonUpper = [1]
        canDynamaxUpper = [1]
        dynamaxTurnUpper = [1]
        teamTypesUpper = [1]*12
        teamDmgMultiplyerUpper = [1]*6
        activePokemonMovesStatusEffectsUpper = [1]*4
        teamHealthUpper = [1]*6
        activePokemonSideConditionsUpper = [1]*20
        activePokemonStatusUpper = [1]
        activePokemonStatsUpper = [1]*6
        activePokemonAbilityUpper = [1]*1
        activePokemonStatusCounterUpper = [1]*1
        activePokemonMovesAccuracyUpper = [1]*4
        activePokemonMovesCritRatioUpper = [1]*4
        activePokemonMovesCurrentPpUpper = [1]*4
        activePokemonMovesExpectedHitsUpper = [1]*4
        activePokemonMovesForceSwitchUpper = [1]*4
        activePokemonMovesHealsUpper = [1]*4
        activePokemonMovesPriorityUpper = [1]*4
        activePokemonMovesRecoilUpper = [1]*4
        activePokemonMovesSideConditionsUpper = [1]*4
        activePokemonMovesTerrainUpper = [1]*4
        activePokemonMovesWeatherUpper = [1]*4
        faintedOpponentTeamPokemonUpper = [1]
        opponentTeamTypesUpper = [1]*12
        opponentDyanamaxTurnUpper = [1]
        opponentCanDynamaxUpper = [1]
        opponentSideConditionsUpper = [1]*20
        opponentTeamHealthUpper = [1]*6
        activeOpponentPokemonStatusUpper = [1]
        activeOpponentPokemonAbilityUpper = [1]*1
        activeFieldsUpper = [0]*12
        currentWeatherUpper = [1]

        inputVectorLow =np.concatenate( 
        [
            activePokemonMovesBasePowerLower,
            activePokemonMovesDmgMultiplierLower,
            faintedTeamPokemonLower,
            canDynamaxLower,
            dynamaxTurnLower,
            teamTypesLower,
            teamDmgMultiplyerLower,
            activePokemonMovesStatusEffectsLower,
            teamHealthLower,
            activePokemonSideConditionsLower,
            activePokemonStatusLower,
            activePokemonStatsLower,
            activePokemonAbilityLower,
            activePokemonStatusCounterLower,
            activePokemonMovesAccuracyLower,
            activePokemonMovesCritRatioLower,
            activePokemonMovesCurrentPpLower,
            activePokemonMovesExpectedHitsLower,
            activePokemonMovesForceSwitchLower,
            activePokemonMovesHealsLower,
            activePokemonMovesPriorityLower,
            activePokemonMovesRecoilLower,
            activePokemonMovesSideConditionsLower,
            activePokemonMovesTerrainLower,
            activePokemonMovesWeatherLower,
            faintedOpponentTeamPokemonLower,
            opponentTeamTypesLower,
            opponentDyanamaxTurnLower,
            opponentCanDynamaxLower,
            opponentSideConditionsLower,
            opponentTeamHealthLower,
            activeOpponentPokemonStatusLower,
            activeOpponentPokemonAbilityLower,
            activeFieldsLower,
            currentWeatherLower,
        ] )

        inputVectorHigh = np.concatenate (
            [
                activePokemonMovesBasePowerUpper,
                activePokemonMovesDmgMultiplierUpper,
                faintedTeamPokemonUpper,
                canDynamaxUpper,
                dynamaxTurnUpper,
                teamTypesUpper,
                teamDmgMultiplyerUpper,
                activePokemonMovesStatusEffectsUpper,
                teamHealthUpper,
                activePokemonSideConditionsUpper,
                activePokemonStatusUpper,
                activePokemonStatsUpper,
                activePokemonAbilityUpper,
                activePokemonStatusCounterUpper,
                activePokemonMovesAccuracyUpper,
                activePokemonMovesCritRatioUpper,
                activePokemonMovesCurrentPpUpper,
                activePokemonMovesExpectedHitsUpper,
                activePokemonMovesForceSwitchUpper,
                activePokemonMovesHealsUpper,
                activePokemonMovesPriorityUpper,
                activePokemonMovesRecoilUpper,
                activePokemonMovesSideConditionsUpper,
                activePokemonMovesTerrainUpper,
                activePokemonMovesWeatherUpper,
                faintedOpponentTeamPokemonUpper,
                opponentTeamTypesUpper,
                opponentDyanamaxTurnUpper,
                opponentCanDynamaxUpper,
                opponentSideConditionsUpper,
                opponentTeamHealthUpper,
                activeOpponentPokemonStatusUpper,
                activeOpponentPokemonAbilityUpper,
                activeFieldsUpper,
                currentWeatherUpper,
            ]
            )

        return Box(
            np.array(inputVectorLow, dtype=np.float32),
            np.array(inputVectorHigh, dtype=np.float32),
            dtype=np.float32,
        )