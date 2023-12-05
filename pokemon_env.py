from pyboy import PyBoy
from battle_wrapper import BattleWrapper
import random

"""
An environment to represent Pokemon battles, to be used with AI methods
(Mirrors an OpenAI Gym environment)
"""
class PokemonEnvironment():

    def __init__(self, pyboy, wrapper):
        self.pyboy = pyboy
        self.wrapper = wrapper
        self.num_turns = 0
        # initialize the emulator
        pyboy.tick()
        # change settings to speed up the game
        #pyboy.set_emulation_speed(0)
        pyboy.set_memory_value(0xD355, 0b11000000)

    def _get_state(self):
        """
        STATE INFORMATION:
            - Player Pokemon Type 1
            - Player Pokemon Type 2
            - Player Pokemon Current/Max HP %
            - Player Pokemon Status
            - Player Pokemon Move 1 Type
            - Player Pokemon Move 1 Power
            - Player Pokemon Move 1 Accuracy
            - Player Pokemon Move 2 Type
            - Player Pokemon Move 2 Power
            - Player Pokemon Move 2 Accuracy
            - Player Pokemon Move 3 Type
            - Player Pokemon Move 3 Power
            - Player Pokemon Move 3 Accuracy
            - Player Pokemon Move 4 Type
            - Player Pokemon Move 4 Power
            - Player Pokemon Move 4 Accuracy
            - Enemy Pokemon Type 1
            - Enemy Pokemon Type 2
            - Enemy Pokemon Current/Max HP %
            - Enemy Pokemon Status

        NOT INCLUDED:
            - Enemy Move 1-4
            - Player/Enemy Attack/Def/Speed/Spec stats
            - Player/Enemy Level Stats
            - Player/Enemy Move PP

        Returns:
            - 1: the features of the state
            - 2: a list of legal actions at that state
        """
        pass

    def _get_reward(self, params):
        """
        PARAMS:
            - 'player_pokemon_remaining': Amount of previous state.
            - 'enemy_pokemon_remaining': Amount of previous state
        REWARD INFORMATION:
            - If the player's pokemon has been defeated, add a reward of -10
            - If the opponent's pokemon has been defeated, add a reward of +10
            - If neither has happened, add a reward of 0
            - On each turn, give a reward of -1
        """
        reward = -1
        # check for changes in pokemon remaining for either side
        if params['player_pokemon_remaining'] - self.wrapper.get_player_pokemon_remaining != 0:
            reward -= 10
        if params['enemy_pokemon_remaining'] - self.wrapper.get_enemy_pokemon_remaining != 0:
            reward += 10
        return reward

    def reset(self, seed=-1, starting=0):
        self.num_turns = 0
        self.wrapper = BattleWrapper(self.pyboy)
        # if starting pokemon is specified, load it
        if 1 <= starting <= 6:
            file_like_object = open(f'roms/states/champ_battle_begin_{starting}.state', 'rb')
            self.pyboy.load_state(file_like_object)
            return self._get_state()
        if seed != -1:
            random.seed(seed)
        index = random.randint(1,6)
        file_like_object = open(f'roms/states/champ_battle_begin_{index}.state', 'rb')
        self.pyboy.load_state(file_like_object)
        return self._get_state()

    def step(self, action):
        """
        Parameters
        ---------
        action: action to be taken


        Returns
        ---------
        next_state: state after the step is taken
        reward: reward after the step is taken
        is_terminal: true if the battle has ended, false otherwise
        """
        # before acting, store current and enemy pokemon remaining in a dict for reward
        pokemon_remaining = {
            "player_pokemon_remaining": self.wrapper.get_player_pokemon_remaining(),
            "enemy_pokemon_remaining": self.wrapper.get_enemy_pokemon_remaining()
        }
        self.wrapper.act(action)
        self.num_turns += 1
        return self._get_state(), self._get_reward({}), self.wrapper.is_battle_over() != 0


