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

    def __init__(self, p_transition, p_emission):
        '''
        Hidden Markov Model

        input:
            p_transition: initial transition probabilites, np.array of dimensions (n_hidden_states + 1) * (n_hidden_states + 1) (+1 is because of START/STOP state)
            p_emission: initial emission probabilities, np.array of dimensions n_observation_classes * n_hidden_states

        n_hidden_states is the number of different classes of hidden states
        n_observation_classes is the number of different classes of observations
        '''

        self.p_transition = p_transition
        self.p_emission = p_emission
        self.n_hidden_states = p_transition.shape[0] - 1
        self.n_observation_classes = p_emission.shape[0]

    def train(self, observations, n_iterations):
        '''
        input:
            observations: sequence of observations, 1D np.array of integers
            n_iterations: number of EM iterations, integer
        '''

        self.observations = np.asarray(observations)
        self.n_observations = len(observations)

        # convert observation datapoints to indices starting from 0
        self.observation_indices = self.renumber_observations()

        # all unique observation labels
        self.observation_labels = list(np.unique(observations))
        assert len(self.observation_labels) <= self.n_observation_classes

        print('Training a HMM with {} hidden states and {} observation classes, using {} observation data points'.format(self.n_hidden_states, self.n_observation_classes, self.n_observations))

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
        self.calc_forward_probabilities()
        self.calc_backward_probabilities()
        self.calc_gamma()
        self.calc_delta()

    def maximization(self):
        '''
        maximization step of Baum-Welch algorithm
        '''
        pass

    def calc_forward_probabilities(self):
        '''
        calculate forward probabilities (alpha)
        alpha has dimensions: self.n_observations * self.n_hidden_states
        '''

        # alpha_i(1) = pi_i * b_i(y_1)
        # P(x|START) probabilities are in final column of transition matrix
        self.forward[0] = self.p_transition[:-1, -1] * self.p_emission[self.observation_indices[0]]

        print(self.forward[0])
        # alpha_i(t+1) = b_i(y_{t+1}) * sum_{j=1}^N alpha_j(t) * a_ji
        # don't use final row and column of p_transition because these are for STOP and START states
        for t in range(1, self.n_observations):
            self.forward[t] = self.p_emission[self.observation_indices[t]] * np.dot(self.p_transition[:-1, :-1], self.forward[t-1])
            print(self.observation_indices[t], self.forward[t])

    def calc_backward_probabilities(self):
        '''
        calculate backward probabilities (beta)
        beta has dimensions: self.n_observations * self.n_hidden_states
        '''

        # beta_i(T) = 1
        self.backward[self.n_observations-1] = 1.0

        # beta_i(t) = sum_{j=1}^N beta_j(t+1) * a_ij * b_j(y_{t+1})
        for t in range(self.n_observations-2, -1, -1):
            self.backward[t] = np.sum(self.backward[t+1] * self.p_emission[self.observation_indices[t+1]] * np.transpose(self.p_transition[:-1, :-1]), axis=1)
            print(self.backward[t])

    def calc_gamma(self):
        '''
        calculate gamma from forward and backward probabilites
        '''
        # don't include last observation because this is the terminal step,
        # therefore no transition to estimate
        for t in range(self.n_observations - 1):
            for i in range(self.n_hidden_states):
                for j in range(self.n_hidden_states):
                    self.gamma[t,i,j] = self.forward[t,i] * self.p_transition[j,i] * self.backward[t+1,j] * self.p_emission[self.observation_indices[t+1],j]
            denom = np.dot(self.forward[t], self.backward[t])
            self.gamma[t,:,:] /= denom
            print(self.gamma[t,:,:])

    def renumber_observations(self):
        '''
        we want to use observation data points directly as indices of the emission matrix
        returns set of observation data points numbered starting at 0
        '''

        return self.observations - min(self.observations)
