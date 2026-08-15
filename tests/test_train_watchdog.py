import unittest

import numpy as np

from RL_train import validate_policy_weights, validate_train_result


ITERATION_210 = {
    "training_iteration": 210,
    "num_env_steps_sampled": 8089200,
    "num_env_steps_sampled_this_iter": 38520,
    "num_env_steps_trained": 8089200,
    "num_env_steps_trained_this_iter": 38520,
    "num_agent_steps_sampled": 48535200,
    "num_agent_steps_trained": 48535200,
    "episodes_this_iter": 37,
    "num_healthy_workers": 3,
    "num_remote_worker_restarts": 0,
    "num_faulty_episodes": 0,
}

ITERATION_211 = {
    "training_iteration": 211,
    "num_env_steps_sampled": 8102040,
    "num_env_steps_sampled_this_iter": 12840,
    "num_env_steps_trained": 8102040,
    "num_env_steps_trained_this_iter": 12840,
    "num_agent_steps_sampled": 48612240,
    "num_agent_steps_trained": 48612240,
    "episodes_this_iter": 0,
    "num_healthy_workers": 2,
    "num_remote_worker_restarts": 2,
    "num_faulty_episodes": 0,
}


class TrainResultWatchdogTest(unittest.TestCase):
    def test_accepts_iteration_210(self):
        validate_train_result(ITERATION_210, 38520, 3)

    def test_rejects_iteration_211(self):
        with self.assertRaisesRegex(RuntimeError, "iteracao 211"):
            validate_train_result(ITERATION_211, 38520, 3)

    def test_rejects_missing_key(self):
        result = ITERATION_210.copy()
        del result["num_healthy_workers"]

        with self.assertRaisesRegex(RuntimeError, "num_healthy_workers ausente"):
            validate_train_result(result, 38520, 3)

    def test_rejects_reset_counters(self):
        result = ITERATION_210.copy()
        result["num_env_steps_sampled"] = 38520
        result["num_env_steps_trained"] = 38520

        with self.assertRaisesRegex(RuntimeError, "esperado 8089200"):
            validate_train_result(result, 38520, 3)

    def test_rejects_non_finite_weights(self):
        weights = {
            "policy_blue": {"layer": np.array([0.0, np.nan])},
            "policy_yellow": {"layer": np.array([0.0, 1.0])},
        }

        with self.assertRaisesRegex(RuntimeError, "policy_blue: 1 pesos NaN/Inf"):
            validate_policy_weights(weights)


if __name__ == "__main__":
    unittest.main()