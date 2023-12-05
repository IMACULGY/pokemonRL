import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class Q_NN(nn.Module):
    def __init__(self, input_dim, dim_hidden_layer, output_dim):
        super(Q_NN, self).__init__()
        # initialize functions
        self.relu = nn.RELU()
        self.elu = nn.ELU()
        self.softmax = nn.Softmax()

        # initialize layers
        self.model = nn.Sequential(
            nn.Linear(input_dim,dim_hidden_layer),
            nn.ReLU(),
            nn.Linear(dim_hidden_layer, dim_hidden_layer),
            nn.ReLU(),
            nn.Linear(dim_hidden_layer, output_dim)
        )


    def forward(self,x):
        value = self.model(x)
        return value

# supervised training for NN with input [X,Y]
def TrainNN(net, x, y, weights, EPOCHS=10):
    # initialize tensors
    x = torch.tensor(x).to(device)
    y1 = torch.stack([a[0] for a in y],dim=0).to(device)
    y2 = torch.FloatTensor([[a[1]] for a in y]).to(device)
    weights = torch.FloatTensor(weights).to(device)

    # print(weights)

    # define losses and optimizer
    celoss = nn.CrossEntropyLoss()
    mseloss = nn.MSELoss()
    opt = torch.optim.RMSprop(net.parameters(), lr=0.0001)
    sumlosses = 0

    # progress bar!
    bar = tqdm.trange(EPOCHS, desc="epoch")


    for _ in bar:
        opt.zero_grad()

        # forward propagation
        pred = net(x)
        
        # compute loss
        l1 = celoss(y1, pred[0])
        l2 = mseloss(y2, pred[1])
        sumlosses = sum([l1, l2])
        # apply sample weighting
        sumlosses = sumlosses * weights
        sumlosses = sumlosses.mean()

        bar.set_description(f"Loss = {sumlosses.item()}")

        # backward propagation
        sumlosses.backward()
        opt.step()

    return sumlosses.item()

# initialize weights of layer m using Glorot initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

# to represent the replay buffer
class ReplayBuffer(object):
    def __init__(self, buffer_size):
        """Args:
               buffer_size (int): size of the replay buffer
        """
        # total size of the replay buffer
        self.total_size = buffer_size

        # create a list to store the transitions
        self._data_buffer = []
        self._next_idx = 0

    def __len__(self):
        return len(self._data_buffer)

    def add(self, obs, act, reward, next_obs, done):
        # create a tuple
        trans = (obs, act, reward, next_obs, done)

        # interesting implementation
        if self._next_idx >= len(self._data_buffer):
            self._data_buffer.append(trans)
        else:
            self._data_buffer[self._next_idx] = trans

        # increase the index
        self._next_idx = (self._next_idx + 1) % self.total_size

    def _encode_sample(self, indices):
        """ Function to fetch the state, action, reward, next state, and done arrays.
        
            Args:
                indices (list): list contains the index of all sampled transition tuples.
        """
        # lists for transitions
        obs_list, actions_list, rewards_list, next_obs_list, dones_list = [], [], [], [], []

        # collect the data
        for idx in indices:
            # get the single transition
            data = self._data_buffer[idx]
            obs, act, reward, next_obs, d = data
            # store to the list
            obs_list.append(np.array(obs, copy=False))
            actions_list.append(np.array(act, copy=False))
            rewards_list.append(np.array(reward, copy=False))
            next_obs_list.append(np.array(next_obs, copy=False))
            dones_list.append(np.array(d, copy=False))
        # return the sampled batch data as numpy arrays
        return np.array(obs_list), np.array(actions_list), np.array(rewards_list), np.array(next_obs_list), np.array(
            dones_list)

    def sample_batch(self, batch_size):
        """ Args:
                batch_size (int): size of the sampled batch data.
        """
        # sample indices with replaced
        indices = [np.random.randint(0, len(self._data_buffer)) for _ in range(batch_size)]
        return self._encode_sample(indices)

# to represent a schedule for epsilon
class LinearSchedule(object):
    """ This schedule returns the value linearly"""
    def __init__(self, start_value, end_value, duration):
        # start value
        self._start_value = start_value
        # end value
        self._end_value = end_value
        # time steps that value changes from the start value to the end value
        self._duration = duration
        # difference between the start value and the end value
        self._schedule_amount = end_value - start_value

    def get_value(self, time):
        # logic: if time > duration, use the end value, else use the scheduled value
        if time > self._duration:
          return self._end_value

        return self._start_value + time/self._duration * self._schedule_amount

# to represent the agent for DQN
class DQNAgent(object):
    # initialize the agent
    def __init__(self,
                 params, load=False, loadPath = 'q_model.pt'
                 ):
        # save the parameters
        self.params = params

        self.device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # environment parameters
        self.action_dim = params['action_dim']
        self.obs_dim = params['observation_dim']

        # executable actions
        self.action_space = params['action_space']

        # create value network
        self.behavior_policy_net = Q_NN().to(device).to(torch.float)
        # create target network
        self.target_policy_net = Q_NN().to(device).to(torch.float)
        # initialize target network with behavior network
        if (load):
            print(f"Loading from {loadPath}")
            self.behavior_policy_net.load_state_dict(torch.load(loadPath))
        else:
            print("Initializing weights...")
            self.behavior_policy_net.apply(init_weights)
        self.behavior_policy_net.apply(init_weights)
        self.target_policy_net.load_state_dict(self.behavior_policy_net.state_dict())

        # optimizer
        self.optimizer = torch.optim.RMSprop(self.behavior_policy_net.parameters(), lr=params['learning_rate'])

    # get action
    def get_action(self, obs, eps):
        if np.random.random() < eps:  # with probability eps, the agent selects a random action
            action = np.random.choice(self.action_space, 1)[0]
            return action
        else:  # with probability 1 - eps, the agent selects a greedy policy
            obs = self._arr_to_tensor(obs).view(1, -1)
            with torch.no_grad():
                q_values = self.behavior_policy_net(obs)
                action = q_values.max(dim=1)[1].item()
            return self.action_space[int(action)]

    # update behavior policy
    def update_behavior_policy(self, batch_data):
        # convert batch data to tensor and put them on device
        batch_data_tensor = self._batch_to_tensor(batch_data)

        #print(batch_data_tensor)

        # get the transition data
        obs_tensor = batch_data_tensor['obs']
        actions_tensor = batch_data_tensor['action']
        next_obs_tensor = batch_data_tensor['next_obs']
        rewards_tensor = batch_data_tensor['reward']
        dones_tensor = batch_data_tensor['done']

        # compute the q value estimation using the behavior network
        estimate = self.behavior_policy_net(obs_tensor).gather(1,actions_tensor)
        #print(f"ESTIMATE: {estimate}")

        # compute the TD target using the target network
        # get next state values
        maxvalues = self.target_policy_net(next_obs_tensor).max(1)[0].detach()
        #print(f"MAX: {maxvalues}")

        target = torch.zeros(self.params['batch_size'])
        for i in range(self.params['batch_size']):
          if dones_tensor[i,0] == 0:
            target[i] = maxvalues[i] * self.params['gamma'] + rewards_tensor[i,0].item()
          else:
            target[i] = rewards_tensor[i,0].item()
        #print(target)

        # compute the loss
        mseloss = nn.MSELoss()
        td_loss = mseloss(target, torch.t(estimate))
        #print(td_loss)

        # minimize the loss
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()

        return td_loss.item()

    # update update target policy
    def update_target_policy(self):
        # hard update
        """CODE HERE: 
                Copy the behavior policy network to the target network
        """
        self.target_policy_net.load_state_dict(self.behavior_policy_net.state_dict())

    # auxiliary functions
    def _arr_to_tensor(self, arr):
        arr = np.array(arr)
        arr_tensor = torch.from_numpy(arr).float().to(self.device)
        return arr_tensor

    def _batch_to_tensor(self, batch_data):
        # store the tensor
        batch_data_tensor = {'obs': [], 'action': [], 'reward': [], 'next_obs': [], 'done': []}
        # get the numpy arrays
        obs_arr, action_arr, reward_arr, next_obs_arr, done_arr = batch_data
        # convert to tensors
        batch_data_tensor['obs'] = torch.tensor(obs_arr, dtype=torch.float32).to(self.device)
        batch_data_tensor['action'] = torch.tensor(action_arr).long().view(-1, 1).to(self.device)
        batch_data_tensor['reward'] = torch.tensor(reward_arr, dtype=torch.float32).view(-1, 1).to(self.device)
        batch_data_tensor['next_obs'] = torch.tensor(next_obs_arr, dtype=torch.float32).to(self.device)
        batch_data_tensor['done'] = torch.tensor(done_arr, dtype=torch.float32).view(-1, 1).to(self.device)

        return batch_data_tensor


def train_dqn_agent(env, params):
    # create the DQN agent
    my_agent = DQNAgent(params)

    # create the epsilon-greedy schedule
    my_schedule = LinearSchedule(start_value=params['epsilon_start_value'],
                                 end_value=params['epsilon_end_value'],
                                 duration=params['epsilon_duration'])

    # create the replay buffer
    replay_buffer = ReplayBuffer(params['replay_buffer_size'])

    # training variables
    episode_t = 0
    rewards = []
    train_returns = []
    train_loss = []
    loss = 0

    # reset the environment
    obs, _ = env.reset()

    # start training
    pbar = tqdm.trange(params['total_training_time_step'])
    last_best_return = 0
    for t in pbar:
        # scheduled epsilon at time step t
        eps_t = my_schedule.get_value(t)
        # get one epsilon-greedy action
        action = my_agent.get_action(obs, eps_t)

        # step in the environment
        next_obs, reward, done, _, _ = env.step(action)

        # add to the buffer
        replay_buffer.add(obs, env.action_names.index(action), reward, next_obs, done)
        rewards.append(reward)

        # check termination
        if done:
            # compute the return
            G = 0
            for r in reversed(rewards):
                G = r + params['gamma'] * G

            if G > last_best_return:
                torch.save(my_agent.behavior_policy_net.state_dict(), f"./{params['model_name']}")

            # store the return
            train_returns.append(G)
            episode_idx = len(train_returns)

            # print the information
            pbar.set_description(
                f"Ep={episode_idx} | "
                f"G={np.mean(train_returns[-10:]) if train_returns else 0:.2f} | "
                f"Eps={eps_t}"
            )

            # reset the environment
            episode_t, rewards = 0, []
            obs, _ = env.reset()
        else:
            # increment
            obs = next_obs
            episode_t += 1

        if t > params['start_training_step']:
            # update the behavior model
            if not np.mod(t, params['freq_update_behavior_policy']):
                # sample batch from replay buffer
                newbatch = replay_buffer.sample_batch(params['batch_size'])
                tdloss = my_agent.update_behavior_policy(newbatch)
                train_loss.append(tdloss)

            # update the target model
            if not np.mod(t, params['freq_update_target_policy']):
                my_agent.update_target_policy()

    # save the results
    return train_returns, train_loss