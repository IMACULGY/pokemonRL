from pyboy import PyBoy
from battle_wrapper import BattleWrapper
import random
import multiprocessing

class PokemonEnvironment():
    """
    An environment to represent Pokemon battles, to be used with AI methods
    (Mirrors an OpenAI Gym environment)
    """

    def __init__(self, pyboy, wrapper):
        self.pyboy = pyboy
        self.wrapper = wrapper
        self.num_turns = 0
        self.state_dim = 20
        self.actions_list = list(range(1,11))

    def _init_emulator(self):
        # initialize the emulator
        self.pyboy.tick()
        # change settings to speed up the game
        self.pyboy.set_emulation_speed(0)
        self.pyboy.set_memory_value(0xD355, 0b11000000)

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
            - Player Other Pokemon Type
            - Player Other Pokemon Health
            - Player Other Pokemon Moves

        Returns:
            - 1: the features of the state
            - 2: a list of legal actions at that state
        """
        player_info = self.wrapper.get_player_pokemon_info()
        current_info = self.wrapper.get_player_current_info()
        pokemon_index = current_info['pokemon_index']
        enemy_info = self.wrapper.get_enemy_pokemon_info()
        state = [
            current_info['type1'],
            current_info['type2'],
            current_info['current_hp']/current_info['max_hp'],
            current_info['status'],
            player_info[pokemon_index]['move1'][1],
            player_info[pokemon_index]['move1'][2],
            player_info[pokemon_index]['move1'][3],
            player_info[pokemon_index]['move2'][1],
            player_info[pokemon_index]['move2'][2],
            player_info[pokemon_index]['move2'][3],
            player_info[pokemon_index]['move3'][1],
            player_info[pokemon_index]['move3'][2],
            player_info[pokemon_index]['move3'][3],
            player_info[pokemon_index]['move4'][1],
            player_info[pokemon_index]['move4'][2],
            player_info[pokemon_index]['move4'][3],
            enemy_info['type1'],
            enemy_info['type2'],
            enemy_info['current_hp'] / enemy_info['max_hp'],
            enemy_info['status']
        ]
        legal_actions = self.wrapper.get_available_actions() if self.wrapper.is_battle_over() == 0 else []
        return (state, legal_actions)

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
        if params['player_pokemon_remaining'] - self.wrapper.get_player_pokemon_remaining() != 0:
            reward -= 10
        if params['enemy_pokemon_remaining'] - self.wrapper.get_enemy_pokemon_remaining() != 0:
            reward += 10
        return reward

    def reset(self, seed=-1, starting=0):
        self.num_turns = 0
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
        self.wrapper = BattleWrapper(self.pyboy)
        self._init_emulator()
        return self._get_state()

    def step(self, action, timeout=5):
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
        self.wrapper.act(action, timeout)
        self.num_turns += 1
        return self._get_state(), self._get_reward(pokemon_remaining), self.wrapper.is_battle_over() != 0

    def get_num_turns(self):
        return self.num_turns

class UpdatedPokemonEnvironment(PokemonEnvironment):
    """
    A new environment with a different state representation, which adds features for the status and HP % of the users' other Pokemon.
    With this change, I hope the agent will make more rational decisions when it comes to switching Pokemon.

    Refer to the new __get_state() function to see what was changed.
    """

    def __init__(self, pyboy, wrapper):
        super().__init__(pyboy, wrapper)
        
    def _get_state(self):
        """
        STATE INFORMATION:
            - Player Pokemon Type 1
            - Player Pokemon Type 2
            - Player Pokemon 1 (Pikachu) Current/Max HP %
            - Player Pokemon 1 (Pikachu) Status
            - Player Pokemon 2 (Charizard) Current/Max HP %
            - Player Pokemon 2 (Charizard) Status
            - Player Pokemon 3 (Blastoise) Current/Max HP %
            - Player Pokemon 3 (Blastoise) Status
            - Player Pokemon 4 (Venusaur) Current/Max HP %
            - Player Pokemon 4 (Venusaur) Status
            - Player Pokemon 5 (Pigeot) Current/Max HP %
            - Player Pokemon 5 (Pigeot) Status
            - Player Pokemon 6 (Snorlax) Current/Max HP %
            - Player Pokemon 6 (Snorlax) Status
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
            - Player Other Pokemon Type
            - Player Other Pokemon Moves

        Returns:
            - 1: the features of the state
            - 2: a list of legal actions at that state
        """
        player_info = self.wrapper.get_player_pokemon_info()
        current_info = self.wrapper.get_player_current_info()
        pokemon_index = current_info['pokemon_index']
        enemy_info = self.wrapper.get_enemy_pokemon_info()
        state = [
            current_info['type1'],
            current_info['type2'],
            player_info[self.wrapper.map_to_party(0)]['current_hp'] / player_info[self.wrapper.map_to_party(0)]['max_hp'],
            player_info[self.wrapper.map_to_party(0)]['status'],
            player_info[self.wrapper.map_to_party(1)]['current_hp'] / player_info[self.wrapper.map_to_party(1)]['max_hp'],
            player_info[self.wrapper.map_to_party(1)]['status'],
            player_info[self.wrapper.map_to_party(2)]['current_hp'] / player_info[self.wrapper.map_to_party(2)]['max_hp'],
            player_info[self.wrapper.map_to_party(2)]['status'],
            player_info[self.wrapper.map_to_party(3)]['current_hp'] / player_info[self.wrapper.map_to_party(3)]['max_hp'],
            player_info[self.wrapper.map_to_party(3)]['status'],
            player_info[self.wrapper.map_to_party(4)]['current_hp'] / player_info[self.wrapper.map_to_party(4)]['max_hp'],
            player_info[self.wrapper.map_to_party(4)]['status'],
            player_info[self.wrapper.map_to_party(5)]['current_hp'] / player_info[self.wrapper.map_to_party(5)]['max_hp'],
            player_info[self.wrapper.map_to_party(5)]['status'],
            player_info[pokemon_index]['move1'][1],
            player_info[pokemon_index]['move1'][2],
            player_info[pokemon_index]['move1'][3],
            player_info[pokemon_index]['move2'][1],
            player_info[pokemon_index]['move2'][2],
            player_info[pokemon_index]['move2'][3],
            player_info[pokemon_index]['move3'][1],
            player_info[pokemon_index]['move3'][2],
            player_info[pokemon_index]['move3'][3],
            player_info[pokemon_index]['move4'][1],
            player_info[pokemon_index]['move4'][2],
            player_info[pokemon_index]['move4'][3],
            enemy_info['type1'],
            enemy_info['type2'],
            enemy_info['current_hp'] / enemy_info['max_hp'],
            enemy_info['status']
        ]
        legal_actions = self.wrapper.get_available_actions() if self.wrapper.is_battle_over() == 0 else []
        return (state, legal_actions)
