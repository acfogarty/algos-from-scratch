import numpy as np
from hmm import HMM

# ice-cream/weather example data from 
# 'An Interactive Spreadsheet for Teaching the Forward-Backward Algorithm'
# Jason Eisner
# http://www.cs.jhu.edu/~jason/papers/#eisner-2002-tnlp

# n_hidden_states = 2 (Hot or Cold)
# n_observation_classes = 3 (number of ice creams)

# emission probabilities [p(1|C), p(1|H)], [p(2|C), p(2|H)], [p(3|C), p(3|H)]
p_emission = np.array([[0.7, 0.1], [0.2, 0.2], [0.1, 0.7]])

# transmission probabilities [ [p(C|C), p(C|H), p(C|START)], [p(H|C), p(H|H), p(H|START)], [p(STOP|C), p(STOP|H), p(STOP|START)], ]
p_transmission = np.array([ [0.8, 0.1, 0.5], [0.1, 0.8, 0.5], [0.1, 0.1, 0] ])

# observation data points (number of ice creams)
observations = [2, 3, 3, 2, 3, 2, 3, 2, 2, 3, 1, 3, 3, 1, 1, 1, 2, 1, 1, 1, 3, 1, 2, 1, 1, 1, 2, 3, 3, 2, 3, 2, 2]

n_iterations = 10

model = HMM(p_transmission, p_emission)
model.train(observations, n_iterations)
