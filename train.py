from pokemon_env import PokemonEnvironment
from pyboy import PyBoy
from battle_wrapper import BattleWrapper
from dqn import train_dqn_agent

if __name__ == '__main__':
    # set the random seed
    # np.random.seed(1234)
    # random.seed(1234)
    # torch.manual_seed(1234)

    pyboy = PyBoy('roms/pokemon.gb', cgb=0)
    wrapper = BattleWrapper(pyboy)
    # create environment
    my_env = PokemonEnvironment(pyboy, wrapper)

    # create training parameters
    train_parameters = {
        'observation_dim': 20,
        'action_dim': 10,
        'action_space': list(range(1,11)),
        'hidden_layer_num': 2,
        'hidden_layer_dim': 512,
        'gamma': 0.99,

        'total_training_time_step': 500_000,

        'epsilon_start_value': 1.0,
        'epsilon_end_value': 0.01,
        'epsilon_duration': 250_000,

        'replay_buffer_size': 50000,
        'start_training_step': 500,
        'freq_update_behavior_policy': 4,
        'freq_update_target_policy': 2000,

        'batch_size': 64,
        'learning_rate': 1e-2,

        'model_name': "pokemon_dqn.pt"
    }

    # create experiment
    train_returns, train_loss = train_dqn_agent(my_env, train_parameters)