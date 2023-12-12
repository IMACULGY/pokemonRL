from pyboy import PyBoy
from battle_wrapper import BattleWrapper
import random

# this one is self-explanatory
SUPER_FAST_MODE = True
# turn this on if you want to demo the battle stepper
DEMO_MODE = True
# turn me on if you want to step through the actions manually
# otherwise, we will demo it by executing actions randomly until the battle ends
MANUAL_STEP_MODE = False

"""
Some Demo Functions
"""
def run_manual_stepper():    
    print("Welcome to the battle stepper.")
    while True:
        acts = wrapper.get_available_actions()
        print(acts)
        num = int(input("Input an action from the list above (0 to stop): "))
        if num == 0: break
        elif num == 11:
            print("Gained control for 10 seconds...")
            for i in range(600):
                pyboy.tick()
        elif num == 12:
            print("Ticking once...")
            pyboy.tick() # once, for debugging
        elif num == 13:
            hexval = int(input("Input a memory address to read: "), 16)
            wrapper.debug_print(hexval)
        elif num not in acts:
            print(f"Invalid action {num}. try again")
        else:
            wrapper.act(num)
    pyboy.stop()

def run_random_stepper():
    #print("Welcome to the random battle stepper.")
    while wrapper.is_battle_over() == 0:
        acts = wrapper.get_available_actions()
        moves = []
        swaps = []
        for a in acts:
            if a < 5:
                moves.append(a)
            else:
                swaps.append(a)
        selected_act = 0
        if moves and random.random() < 0.66:
            selected_act = random.choice(moves)
        elif swaps:
            selected_act = random.choice(swaps)
        else:
            selected_act = random.choice(moves)
        #selected_act = random.choice(acts)
        print(f"Selected action: {selected_act}")
        wrapper.act(selected_act)
    print("Battle is over. Huzzah")
    print("You won" if wrapper.is_battle_over() == 1 else "You lost")



## INIT EMULATOR FOR DEMO

pyboy = PyBoy('roms/pokemon.gb', cgb=0)
file_like_object = open('roms/states/champ_battle_begin_2.state', 'rb')
pyboy.load_state(file_like_object)
pyboy.tick()
wrapper = BattleWrapper(pyboy)
print(wrapper.get_player_pokemon_info())

# change settings to speed up the game
if SUPER_FAST_MODE: pyboy.set_emulation_speed(0)
pyboy.set_memory_value(0xD355, 0b11000000)

if not DEMO_MODE:
    while True:
        pyboy.tick()
        #wrapper.debug_print(0xCC29)
        #wrapper.debug_print(0xD059)
        #0xc4b9, one of the text bits
        #0xc4f2, bottom right where the cursor is blinking
        #0xc4fc, very bottom tile - 125 in battle screen, 122 otherwise

if MANUAL_STEP_MODE:
    run_manual_stepper()
else:
    run_random_stepper()
