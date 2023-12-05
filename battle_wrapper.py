from pyboy import PyBoy, WindowEvent
import constants

inputs = {
    'left': [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT],
    'right': [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT],
    'up': [WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP],
    'down': [WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN],
    'a': [WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A],
    'b': [WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B],
}

pokemon_map = [
    'Pikachu',
    'Charizard',
    'Blastoise',
    'Venusaur',
    'Pidgeot',
    'Snorlax'
]

"""
Useful functions for reading data and inputting actions for
Pokemon battles through the emulator.
"""
class BattleWrapper():

    def __init__(self, pyboy):
        self.pyboy = pyboy
        self.enemy_pokemon_remaining = 6
        self.pokemon_party_order = [p['name'] for p in self.get_player_pokemon_info()]

    # print any debugging information
    def debug_print(self, memaddr):
        print(f"{hex(memaddr)}: {self.pyboy.get_memory_value(memaddr)}")

    # Press a button, tick, and release the button
    def press_and_release(self, input):
        self.pyboy.send_input(inputs[input][0])
        self.pyboy.tick()
        self.pyboy.send_input(inputs[input][1])
        for i in range(9):
            self.pyboy.tick()
        #print(f"Pressed and released {input}")

    """
    Perform an action in battle, then tick until the next battle screen.
    Fight:
        1 - Move 1
        2 - Move 2
        3 - Move 3
        4 - Move 4
    Switch
        5 - Pikachu
        6 - Charizard
        7 - Blastoise
        8 - Venusaur
        9 - Pidgeot
        10 - Snorlax
    """
    def act(self, action):
        if 1 <= action <= 4:
            self.press_and_release('a')
            # figure out where the cursor and navigate to the next move
            cursor_spot = self.pyboy.get_memory_value(0xCC2A)
            while cursor_spot != action:
                if cursor_spot < action:
                    self.press_and_release('down')
                    cursor_spot += 1
                else:
                    self.press_and_release('up')
                    cursor_spot -= 1
            self.press_and_release('a')
        elif 5 <= action <= 10:
            # get the correctly-ordered index of the pokemon in the party
            party_index = self.map_to_party(action-5)
            if (self.pyboy.get_memory_value(0xCC29) == 17):
                self.press_and_release('right')
                self.press_and_release('a')
                # wait until the next menu screen
                for i in range(14):
                    self.pyboy.tick()
            # figure out where the cursor is and navigate to the next pokemon
            cursor_spot = self.pyboy.get_memory_value(0xCC2A)
            while cursor_spot != party_index:
                if cursor_spot < party_index:
                    self.press_and_release('down')
                    cursor_spot += 1
                else:
                    self.press_and_release('up')
                    cursor_spot -= 1
            self.press_and_release('a')
            self.press_and_release('a')
            # write a dummy value
            # why didnt i think of this earlier
            # it could have probably saved me hours
            # i am the goat of making the worlds worst solutions to these problems
            self.pyboy.set_memory_value(0xcc29, 0)
            self.progress_to_next_battle_screen([3,17])
        
        self.progress_to_next_battle_screen([3,17])

    """
    Spam A until one of the next screens shows up.
    Screens:
        3 - Switch pokemon screen
        17 - Main battle screen
    """
    def progress_to_next_battle_screen(self, screens):
        current_screen = self.pyboy.get_memory_value(0xCC29)
        # 17 - battle screen
        # 3 - select a pokemon
        tile_bit_value = self.pyboy.get_memory_value(0xc4fc)
        # 125 - battle screen
        # 122 - otherwise
        killed_a_pokemon = False
        while current_screen not in screens or (17 in screens and current_screen == 17 and tile_bit_value != 125):
            self.press_and_release('a')
            current_screen = self.pyboy.get_memory_value(0xCC29)
            tile_bit_value = self.pyboy.get_memory_value(0xc4fc)
            # if the enemy pokemon has fainted, do an update
            if not killed_a_pokemon and self.get_enemy_pokemon_info()['current_hp'] == 0:
                killed_a_pokemon = True
                print("You killed somebody.")
                self.enemy_pokemon_remaining -= 1
            if self.is_battle_over() != 0:
                print("It's over.")
                # you win or lose
                break

        self.pyboy.tick()
        self.pyboy.tick()
        if current_screen == 3:
            # tick a few more times for good measure
            for i in range(6):
                self.pyboy.tick()

    """
    If you won the battle, return 1. If you lost, the battle, return -1. Else, return 0
    """
    def is_battle_over(self):
        if len(self.get_player_pokemon_available()) == 0:
            return -1
        elif self.enemy_pokemon_remaining == 0:
            return 1
        return 0


    def get_player_pokemon_info(self):
        pokemon_info = []
        offsets = [0xD16B, 0xD197, 0xD1C3, 0xD1EF, 0xD21B, 0xD247]
        for o in offsets:
            pokemon_info.append({
                "name": self.get_pokemon_name(self.pyboy.get_memory_value(o + 0)),
                "level": self.pyboy.get_memory_value(o + 33),
                "type1": self.pyboy.get_memory_value(o + 5),
                "type2": self.pyboy.get_memory_value(o + 6),
                "current_hp": self.pyboy.get_memory_value(o + 1) * 256 + self.pyboy.get_memory_value(o + 2),
                "max_hp": self.pyboy.get_memory_value(o + 34) * 256 + self.pyboy.get_memory_value(o + 35),
                "move1": self.get_move_info(self.pyboy.get_memory_value(o + 8)),
                "move2": self.get_move_info(self.pyboy.get_memory_value(o + 9)),
                "move3": self.get_move_info(self.pyboy.get_memory_value(o + 10)),
                "move4": self.get_move_info(self.pyboy.get_memory_value(o + 11)),
            })
        return pokemon_info

    def get_player_current_info(self):
        pokemon_info = self.get_player_pokemon_info()
        pokemon_index = self.pyboy.get_memory_value(0xCC2F)
        player_info = {
            "pokemon_index": pokemon_index,
            "type1": self.pyboy.get_memory_value(0xD019),
            "type2": self.pyboy.get_memory_value(0xD01A),
            "current_hp": self.pyboy.get_memory_value(0xD015) * 256 + self.pyboy.get_memory_value(0xD016),
            "max_hp": pokemon_info[pokemon_index]['max_hp'],
            "status": self.pyboy.get_memory_value(0xD018),
            "pp1": self.pyboy.get_memory_value(0xD02D),
            "pp2": self.pyboy.get_memory_value(0xD02E),
            "pp3": self.pyboy.get_memory_value(0xD02F),
            "pp4": self.pyboy.get_memory_value(0xD030),
        }
        return player_info

    """
    Return a list of the pokemon that haven't died yet.
    """
    def get_player_pokemon_available(self):
        pokemon = []
        pokemon_info = self.get_player_pokemon_info()
        for i,p in enumerate(pokemon_info):
            if p['current_hp'] != 0:
                pokemon.append(i)
        return pokemon
        

    def get_enemy_pokemon_info(self):
        enemy_info = {
            "pokemon_remaining":self.enemy_pokemon_remaining,
            "type1":self.pyboy.get_memory_value(0xCFEA),
            "type2":self.pyboy.get_memory_value(0xCFEB),
            "current_hp":self.pyboy.get_memory_value(0xCFE6) * 256 + self.pyboy.get_memory_value(0xCFE7),
            "max_hp":self.pyboy.get_memory_value(0xCFF4) * 256 + self.pyboy.get_memory_value(0xCFF5),
        }
        return enemy_info

    def get_player_pokemon_remaining(self):
        return len(self.get_player_pokemon_available)

    def get_enemy_pokemon_remaining(self):
        return self.enemy_pokemon_remaining
        
    def get_move_info(self, moveID):
        move_info = [constants.moves_list[moveID-1]]
        move_info.extend(constants.moves[move_info[0]])
        return move_info

    def get_pokemon_name(self, pokemonID):
        return constants.pokemon_list[pokemonID-1]

    # convert an index from the player's party to an index in the pokemon map
    def party_to_map(self, partyIndex):
        return pokemon_map.index(self.pokemon_party_order[partyIndex])

    # convert an index from the pokemon map to its index in the player's party
    def map_to_party(self, mapIndex):
        return self.pokemon_party_order.index(pokemon_map[mapIndex])

    def get_available_actions(self):
        acts = []
        player_info = self.get_player_current_info()
        pokemon = [self.party_to_map(p) + 5 for p in self.get_player_pokemon_available()]
        # if we are not in the select pokemon screen, add moves and remove currently selected pokemon
        if self.pyboy.get_memory_value(0xCC29) != 3:
            pokemon.remove(self.party_to_map(player_info['pokemon_index']) + 5)
            for i in range(4):
                if player_info[f'pp{i+1}'] != 0:
                    acts.append(i+1)
        acts.extend(pokemon)
        return acts

    