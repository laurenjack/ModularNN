from unittest import TestCase
from mock import *

from srcNN.problems.problem_generator import ProblemGenerator
from experiment import ExperimentFactory
from srcNN.problems.problem_domain import ProblemFactory

class ExperimentFactorySpec(TestCase):

    def test_create_single_binary(self):
        obs_fact = ProblemFactory()
        prob_gen_mock = create_autospec(ProblemGenerator)

        obs = obs_fact.create_observation([0,1], [1]);
        mock_train = [obs.as_tuple(), obs.as_tuple()]
        prob_gen_mock.uniform_random_observations = Mock(return_value=mock_train)
        exp_fact = ExperimentFactory(prob_gen_mock)

        result = exp_fact.create_single_binary_observation(obs, 2)

        prob_gen_mock.uniform_random_observations.assert_called_with([obs], 2)

        self.assertEquals([obs], result.base_obs)
        self.assertEquals(mock_train, result.train)

    def test_create_uniform_experiment(self):
            #create test observations
            pg = ProblemGenerator(None)
            two_input_pSet = pg.generate_problems(2)
            """
            0, 0 -> 1
            0, 1 -> 0
            1, 0 -> 1
            1, 1 -> 0
            """
            base_obs = two_input_pSet.baseProblems[10].observations

            #mock the problem generator that Experiment factory is dependant on
            mock_train = [base_obs[2].as_tuple(), base_obs[0].as_tuple(), base_obs[1].as_tuple()]
            prob_gen_mock = create_autospec(ProblemGenerator)
            prob_gen_mock.uniform_random_observations = Mock(return_value=mock_train)
            exp_fact = ExperimentFactory(prob_gen_mock)

            result = exp_fact.create_uniform_experiment(base_obs, 3)

            prob_gen_mock.uniform_random_observations.assert_called_with(base_obs, 3)

            self.assertEquals(base_obs, result.base_obs)
            self.assertEquals(mock_train, result.train)
