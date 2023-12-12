# Pokémon

![image](https://github.com/IMACULGY/pokemonRL/assets/32719081/7aa0dcc8-3801-4059-a1b5-5d63782ba0e9)

Code for my experiments with training a battle agent for the original Pokemon Red. Read the report here (soon).

## How to Run

After getting the source code, grab all the necessary requirements for the project by running the command `pip install -r requirements.txt` in the local project directory.

### Emulator and Game Wrapper

The emulator is provided by the useful module [pyboy](https://github.com/Baekalfen/PyBoy).

The wrapper code for Pokemon Red is written by me, and located in [`battle_wrapper.py`](battle_wrapper.py). The demo for this wrapper is located at [`emu_demo.py`](emu_demo.py). There are some variables at the top of the file that you may change. To fully control the emulator with arrow keys, X, and Z, set `DEMO_MODE` to false. There are two demo modes, a manual stepper and a random stepper. Change which one to use by editing `MANUAL_STEP_MODE`. Be advised, the manual stepper will ask for user input.

Any files for game ROMs, RAMs, and states are located in the [`/roms`](/roms/) directory.

### Model Training

The code for the environment, which mirrors that of an OpenAI gym environment, can be found in [`pokemon_env.py`](pokemon_env.py). The code for the DQN agent, including the neural network, replay buffer, DQN implementation, and training functions is located in [`dqn.py`](dqn.py).

To train a model, run the script [`train.py`](train.py). Note that any changes to the network parameters and dimensions must also be reflected in the environment you choose to use. For example, for my two experiments, I have two seperate environments - PokemonEnvironment and UpdatedPokemonEnvironment (aptly named).

The model training function will output three numpy arrays and multiple PyTorch network files along the way, including temporary, final, and best network (which generated the highest return in an episode).

### Experiments

To demo my experiments, run the script [`dqn_demo.py`](dqn_demo.py). There are a few options available which can be toggled by changing some variables. When enabling `TRIAL_MODE`, a trial of 100 runs of the battle will be conducted at the fastest possible speed (the same speed at which the model was trained). Data for total wins, losses, and returns will be output at the end of the trial. `MODEL_NUM` lets you specify which of my models will be tested, 1 or 2.

The results from my two experiments are located in the notebook file, [run_experiments_plots.ipynb](run_experiments_plots.ipynb).

Any data generated from my experiments is located in the [`/data`](/data/) folder.

## Gallery
