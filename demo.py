from pyboy import PyBoy
from battle_wrapper import BattleWrapper

# turn this on if you want to step through actions one at a time
DEMO_MODE = True

pyboy = PyBoy('roms/pokemon.gb', cgb=0)
wrapper = BattleWrapper(pyboy)
file_like_object = open('roms/states/champ_battle_begin.state', 'rb')
pyboy.load_state(file_like_object)
pyboy.tick()
print(wrapper.get_player_pokemon_info())
# change settings to speed up the game
pyboy.set_memory_value(0xD355, 0b11000000)
if (not DEMO_MODE):
    while True:
        pyboy.tick()
        

while True:
    num = int(input("Input an action 1-10 (0 to stop): "))
    if num == 0: break
    if num == 11:
        print("Gained control for 10 seconds...")
        for i in range(600):
            pyboy.tick()
    if num == 12:
        print("Ticking once...")
        pyboy.tick() # once, for debugging
    wrapper.act(num)
pyboy.stop()