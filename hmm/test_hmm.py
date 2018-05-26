import numpy as np
from hmm import HMM
import unittest


class HMMTest(unittest.TestCase):


    def test_viterbi(self):

        """
        example data from
        http://blog.ivank.net/viterbi-algorithm-clarified.html
        
        n_hidden_states = 2 (Healthy or Fever)
        n_observation_classes = 3 (Normal, Cold, Dizzy)
        """

        # emission probabilities [p(N|H), p(N|F)], [p(C|H), p(C|F)], [p(D|H), p(D|F)]
        p_emission = np.array([[0.5, 0.1], [0.4, 0.3], [0.1, 0.6]])
        
        # transmission probabilities [ [p(H|H), p(H|F), p(H|START)], 
        #                              [p(F|H), p(F|F), p(F|START)], 
        #                              [p(STOP|H), p(STOP|F), p(STOP|STHRT)] ]
        p_transmission = np.array([ [0.7, 0.4, 0.6], [0.3, 0.6, 0.4], [0.0, 0.0, 0] ])
        
        # observation data points (number of ice creams)
        observations = [0, 1, 2]  # N, C, D
        
        model = HMM(p_transmission, p_emission)
        most_likely_hidden_sequence = model.viterbi(observations)

        reference_answer = [0, 0, 1]  # H, H, F

        self.assertListEqual(list(most_likely_hidden_sequence), reference_answer)

    def test_baum_welch(self):

        pass


unittest.main()
