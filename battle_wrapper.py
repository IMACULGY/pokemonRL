from pyboy import PyBoy, WindowEvent

inputs = {
    'left': [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT],
    'right': [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT],
    'up': [WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP],
    'down': [WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN],
    'a': [WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A],
    'b': [WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B],
}

"""
Useful functions for reading data and inputting actions for
Pokemon battles through the emulator.
"""
class BattleWrapper():

    def __init__(self, pyboy):
        self.pyboy = pyboy

    # Tick by one frame and print any debugging information
    def debug_tick(self):
        self.pyboy.tick()
        # do debugging things

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
        5 - Pokemon 1
        6 - Pokemon 2
        7 - Pokemon 3
        8 - Pokemon 4
        9 - Pokemon 5
        10 - Pokemon 6
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
            if (self.pyboy.get_memory_value(0xCC29) == 17):
                self.press_and_release('right')
                self.press_and_release('a')
                # wait until the next menu screen
                for i in range(14):
                    self.pyboy.tick()
            # figure out where the cursor is and navigate to the next pokemon
            cursor_spot = self.pyboy.get_memory_value(0xCC2A)
            while cursor_spot != action - 5:
                if cursor_spot < action - 5:
                    self.press_and_release('down')
                    cursor_spot += 1
                else:
                    self.press_and_release('up')
                    cursor_spot -= 1
            self.press_and_release('a')
            self.press_and_release('a')
            self.progress_to_next_battle_screen([17])
        
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
        while current_screen not in screens:
            self.press_and_release('a')
            current_screen = self.pyboy.get_memory_value(0xCC29)
        # tick one more time for good measure
        self.pyboy.tick()

    