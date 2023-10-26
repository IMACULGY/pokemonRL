from pyboy import PyBoy
pyboy = PyBoy('roms/pokemon.gb')
while not pyboy.tick():
    pass
pyboy.stop()