import numpy as np

# alpha: forward probabilities (from initial estimate of the hidden state at the first data observation)
# beta: backward probabilities (conditional probability from the final data observation)
# gamma: estimate of state transition for each data observation (combine forward and backward probabilities)
# delta: estimate of hidden state for each data observation (sum gamma function across all transitions)


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

        for i in range(self.n_iterations):
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
