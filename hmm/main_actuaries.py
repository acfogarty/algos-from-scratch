import numpy as np
from hmm import HMM

# chronic disease and medication data example from
# Hidden Markov Models and You, Part Two
# Society of Actuaries, Forecasting and Futurism, Dec 2013

# n_hidden_states = 2 (well or sick), W=1, S=2
# n_observation_classes = 3 (number of medical claims per month)

# emission probabilities [p(0|W), p(0|S)], [p(1|W), p(1|S)], [p(2|W), p(2|S)]
p_emission = np.array([[0.4, 0.2], [0.4, 0.2], [0.2, 0.6]])

# transmission probabilities [ [p(W|W), p(W|S), p(W|START)], [p(S|W), p(S|S), p(S|START)], [p(STOP|W), p(STOP|S), p(STOP|START)] ]
p_transmission = np.array([ [0.7, 0.5, 0.5], [0.3, 0.5, 0.5], [0.0, 0.0, 0.0] ])

# observation data points (number of ice creams)
observations = [2, 1, 0, 1, 2, 0, 1, 1, 0, 1, 1, 2, 0, 0, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2 ]

n_iterations = 10

model = HMM(p_transmission, p_emission)
model.train(observations, n_iterations)
