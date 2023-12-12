from pokemon_env import PokemonEnvironment, UpdatedPokemonEnvironment
from pyboy import PyBoy
from battle_wrapper import BattleWrapper
from dqn import train_dqn_agent
import numpy as np

if __name__ == '__main__':
    # set the random seed
    # np.random.seed(1234)
    # random.seed(1234)
    # torch.manual_seed(1234)

    pyboy = PyBoy('roms/pokemon.gb', cgb=0)
    wrapper = BattleWrapper(pyboy)
    # create environment
    my_env = UpdatedPokemonEnvironment(pyboy, wrapper)

    # create training parameters
    train_parameters = {
        'observation_dim': 30,
        'action_dim': 10,
        'action_space': list(range(1,11)),
        'hidden_layer_num': 2,
        'hidden_layer_dim': 512,
        'gamma': 0.99,

        'total_training_time_step': 350_000,

        'epsilon_start_value': 1.0,
        'epsilon_end_value': 0.01,
        'epsilon_duration': 175_000,

        'replay_buffer_size': 50000,
        'start_training_step': 1000,
        'freq_update_behavior_policy': 4,
        'freq_update_target_policy': 2000,

        'batch_size': 64,
        'learning_rate': 1e-3,

        'model_name': "pokemon_dqn_new.pt"
    }

    # create experiment - good luck
    train_returns, train_loss, turns_taken_per_ep = train_dqn_agent(my_env, train_parameters)
    np.save('data/train_returns_new.npy', train_returns)
    np.save('data/train_loss_new.npy', train_loss)
    np.save('data/turns_taken_per_ep_new.npy', turns_taken_per_ep)