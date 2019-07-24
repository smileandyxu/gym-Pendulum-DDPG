game_name = 'Pendulum-v0'
states_dim = 3 # observation_space dim
action_dim = 1 # action_space dim
hidden_dim_pi1 = 64
hidden_dim_pi2 = 64
hidden_dim_v1 = 32
hidden_dim_v2 = 32

learning_rate_a = 0.001
learning_rate_c = 0.001
initial_var = 3.0
gamma = 0.9        # rationality ratio
tau = 0.01         # update amplitude
decay = 0.995      # exploration decay
max_episode = 500  # episode num
max_step = 200     # step limitation
max_memsize = 8192 # numbers of replay data
batch_size = 32    # train batch size
