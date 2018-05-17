import numpy as np

# #########
# ## HMM ##
# #########

# X_t = discrete hidden variable with N distinct values
# P(X_t | X_{t-1}) is independent of t

# time-independent transition probabilites (N * N matrix):
# A = {a_ij} = P(X_t = j | X_{t-1} = i)

# initial distribution
# pi_i = P(X_1 = i)

# Y_t = observation date points with K distinct values

# emission probabilities (N * K matrix)
# B = {b_j(y_i)} = P(Y_t = y_i | X_t = j)

# theta = (A, B, pi) = (p_transition, p_emission, p_initial)

# ##########################
# ## Baum-Welch algorithm ##
# ##########################

# Finds a local maximum for argmax_theta P(Y | theta)

# alpha: forward probabilities, the probability of seeing y1, y2, ... , yt and being in state i at time t  (calculated recursively from initial estimate of the hidden state at the first data observation)

# alpha_i(t) = P(Y1 = y1, ... Yt = yt, Xt = i | theta)

# alpha_i(1) = pi_i * b_i(y_1)
# alpha_i(t+1) = b_i(y_{t+1}) * sum_{j=1}^N alpha_j(t) * a_ji

# beta: backward probabilities, the probability of the ending partial sequence y_{t+1},...,y_{T} given starting state i at time t (calculated as conditional probability from the final data observation)

# beta_i(t) = P(Y_{t+1} = y_{t+1}, ... Y_T = y_T | X_t = i, theta)

# beta_i(T) = 1
# beta_i(t) = sum_{j=1}^N beta_j(t+1) * a_ij * b_j(y_{t+1})

# gamma: estimate of probability of being in i at t and j at t+1 given the observed sequence and parameters (combine forward and backward probabilities)

# gamma_ij(t) = P(X_t=i, X_{t+1}=j | Y, theta) = alpha_i(t) * a_ij * beta_j(t+1) * b_j(y_{t+1}) / (sum over i and j)


class HMM():

    def __init__(self, p_transmission, p_emission):
        '''
        Hidden Markov Model

        input:
            p_transmission: initial transmission probabilites, np.array of dimensions n_hidden_states * n_hidden_states
            p_emission: initial emission probabilities, np.array of dimensions n_observation_classes * n_hidden_states

        n_hidden_states is the number of different classes of hidden states
        n_observations_classes is the number of different classes of observations
        '''

        self.p_transmission = p_transmission
        self.p_emission = p_emission
        self.n_hidden_states = p_transmission.shape[0]
        self.n_observation_classes = p_emission.shape[0]

    def train(self, observations, n_iterations):
        '''
        input:
            observations: sequence of observations, 1D np.array of integers
            n_iterations: number of EM iterations, integer
        '''

        self.observations = observations
        self.n_observations = len(observations)

        # all unique observation labels
        self.observation_labels = list(np.unique(observations))
        assert len(self.observation_labels) <= self.n_observation_classes

        # initialise
        self.forward = np.zeros((self.n_observations, self.n_hidden_states), dtype=np.float64)
        self.backward = np.zeros((self.n_observations, self.n_hidden_states), dtype=np.float64)
        self.gamma = np.zeros((self.n_observations, self.n_hidden_states, self.n_hidden_states), dtype=np.float64)
        self.delta = np.zeros((self.n_observations, self.n_hidden_states), dtype=np.float64)

        for i in range(n_iterations):
            print('Iteration ', i)
            self.expectation()
            self.maximization()

    def expectation(self):
        '''
        expectation step of Baum-Welch algorithm
        '''
        pass

    def maximization(self):
        '''
        maximization step of Baum-Welch algorithm
        '''
        pass
