from pokemon_env import PokemonEnvironment, UpdatedPokemonEnvironment
from pyboy import PyBoy
from battle_wrapper import BattleWrapper
from dqn import DQNAgent
import numpy as np
import tqdm
"""
Demo the model by running through a battle with the DQN agent
"""

# If set to true, run 100 experiments and count number of wins
# If set to false, only run 1 experiment to see which actions are chosen and rewards are given
TRIAL_MODE=False

# Run model/experiment 1 or 2??
MODEL_NUM=1

if __name__ == '__main__':
    # filename to load model from
    filename = f'data/Experiment{MODEL_NUM}/pokemon_dqn.pt'

    pyboy = PyBoy('roms/pokemon.gb', cgb=0)
    wrapper = BattleWrapper(pyboy)
    # create environment
    my_env = PokemonEnvironment(pyboy, wrapper) if MODEL_NUM == 1 else UpdatedPokemonEnvironment(pyboy, wrapper)

    # create training parameters
    parameters = {
        'observation_dim': 20 if MODEL_NUM == 1 else 30,
        'action_dim': 10,
        'action_space': list(range(1,11)),
        'hidden_layer_num': 2,
        'hidden_layer_dim': 512,
        'gamma': 0.99,
        'learning_rate': 1e-3
    }

    myagent = DQNAgent(parameters, load=True, loadPath=filename)

    if not TRIAL_MODE:
        state = my_env.reset()
        reward = 0
        rewards = []
        done = 0
        pyboy.set_emulation_speed(2)

        # loop until the battle is over
        while done == 0:
            act = myagent.get_action(state, 0.01)
            print(f"Action taken: {act}")
            state, reward, done = my_env.step(act, timeout=60)
            rewards.append(reward)
            print(f"Reward given: {reward}\n")
        
        winner = "won" if my_env.wrapper.is_battle_over() == 1 else "lost"
        print(f"Battle is over. You {winner}.")

        # compute the return
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G
        print(f"Total return: {G}")
    else:
        numWins = 0
        returns = []
        returnsWins = []
        returnsLosses = []
        pbar = tqdm.trange(100)
        for i in pbar:
            pbar.set_description(f"Battle #{i+1}, Wins: {numWins}, Losses: {i-numWins}")

            state = my_env.reset()
            reward = 0
            rewards = []
            done = 0
            pyboy.set_emulation_speed(0)

            # loop until the battle is over
            while done == 0:
                act = myagent.get_action(state, 0.01)
                try:
                    state, reward, done = my_env.step(act, timeout=5)
                except:
                    state = my_env.reset()
                    reward = 0
                    rewards = []
                    done = 0
                else:
                    rewards.append(reward)

            # compute the return
            G = 0
            for r in reversed(rewards):
                G = r + 0.99 * G
            returns.append(G)

            if my_env.wrapper.is_battle_over() == 1:
                numWins += 1
                returnsWins.append(G)
            else:
                returnsLosses.append(G)
        
        print(f"Total wins: {numWins}")
        print(f"Returns: {returns}")
        print(f"Returns in Wins: {returnsWins}")
        print(f"Returns in Losses: {returnsLosses}")


