import importlib.util
import sys
import types
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Os testes de contrato nao precisam dos bindings para validar a matematica.
for module_name in (
    "grSim_Packet_pb2", "grSim_Commands_pb2", "ssl_vision_wrapper_pb2",
    "ssl_vision_detection_pb2", "ssl_vision_geometry_pb2",
):
    sys.modules.setdefault(module_name, types.ModuleType(module_name))

spec = importlib.util.spec_from_file_location(
    "deploy_policy_grsim", ROOT / "deploy_policy_grsim.py"
)
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)

from scripts.model.action_dists_inferece import InferenceBetaDist
from sim2real.state_to_obs import frame_to_observations
from scripts.validate_grsim_deterministic import Case, angle_delta, evaluate


class GrSimActionContractTest(unittest.TestCase):
    def test_global_to_local_at_cardinal_headings(self):
        tangent, normal, angular, kick = deploy.normalized_action_to_grsim(
            [1.0, 0.0, 0.5, 0.25], 90.0
        )
        self.assertAlmostEqual(tangent, 0.0, places=7)
        self.assertAlmostEqual(normal, -1.5, places=7)
        self.assertAlmostEqual(angular, 5.0)
        self.assertAlmostEqual(kick, 3.0)

    def test_translation_is_clipped_and_negative_kick_is_zero(self):
        tangent, normal, _, kick = deploy.normalized_action_to_grsim(
            [1.0, 1.0, 0.0, -1.0], 0.0
        )
        self.assertLessEqual(np.hypot(tangent, normal), 1.5 + 1e-12)
        self.assertEqual(kick, 0.0)

    def test_deterministic_gate_accepts_expected_local_motion(self):
        case = Case("normal_h90", "blue", 0, 90.0, normal=0.5)
        passed, _ = evaluate(case, {
            "vx": -0.5, "vy": 0.0, "omega": 0.0, "speed": 0.5
        })
        self.assertTrue(passed)

    def test_deterministic_gate_rejects_reversed_motion(self):
        case = Case("tangent_h0", "blue", 0, 0.0, tangent=0.5)
        passed, _ = evaluate(case, {
            "vx": -0.5, "vy": 0.0, "omega": 0.0, "speed": 0.5
        })
        self.assertFalse(passed)

    def test_angle_delta_handles_wrap(self):
        delta = angle_delta(np.deg2rad(-179.0), np.deg2rad(179.0))
        self.assertAlmostEqual(np.rad2deg(delta), 2.0)


class ObservationContractTest(unittest.TestCase):
    def setUp(self):
        self.frame = {
            "robots_blue": {
                f"robot_{i}": [-1.5 - 0.2 * i, 0.3 * i, 15.0 * i]
                for i in range(3)
            },
            "robots_yellow": {
                f"robot_{i}": [1.5 + 0.2 * i, -0.3 * i, 180.0 - 15.0 * i]
                for i in range(3)
            },
            "ball": [0.4, -0.2],
        }
        self.actions = {
            **{f"blue_{i}": np.full(4, 0.1 * (i + 1)) for i in range(3)},
            **{f"yellow_{i}": np.full(4, -0.1 * (i + 1)) for i in range(3)},
        }
        self.stacks = {
            **{f"blue_{i}": np.zeros(616) for i in range(3)},
            **{f"yellow_{i}": np.zeros(616) for i in range(3)},
        }

    def test_historical_layout_and_time(self):
        result = frame_to_observations(
            self.frame, self.actions, self.stacks, steps=600
        )
        frame = result["blue_1"][-77:]
        np.testing.assert_allclose(frame[64:68], self.actions["blue_1"])
        self.assertAlmostEqual(frame[76], 0.5)

    def test_reset_stack_contains_only_latest_frame(self):
        result = frame_to_observations(
            self.frame, self.actions, self.stacks, steps=0
        )
        np.testing.assert_array_equal(result["blue_0"][:-77], 0.0)
        self.assertTrue(np.any(result["blue_0"][-77:] != 0.0))

    def test_time_feature_clamped_after_episode_length(self):
        result = frame_to_observations(
            self.frame, self.actions, self.stacks, steps=2400
        )
        self.assertEqual(result["blue_1"][-1], 0.0)


class BetaContractTest(unittest.TestCase):
    def test_deterministic_mean_and_clamp(self):
        logits = torch.tensor([[100.0] * 4 + [-100.0] * 4])
        dist = InferenceBetaDist(logits)
        action = dist.deterministic_sample()
        self.assertTrue(torch.all(torch.isfinite(action)))
        self.assertTrue(torch.all(action <= 1.0))
        self.assertTrue(torch.all(action >= -1.0))

    def test_action_mode_selects_mean_or_sample(self):
        distribution = mock.Mock()
        distribution.deterministic_sample.return_value = torch.tensor([0.25])
        distribution.sample.return_value = torch.tensor([-0.75])

        mean = deploy.select_beta_action(distribution, "mean")
        sample = deploy.select_beta_action(distribution, "sample")

        torch.testing.assert_close(mean, torch.tensor([0.25]))
        torch.testing.assert_close(sample, torch.tensor([-0.75]))

    def test_invalid_action_mode_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "ACTION_MODE invalido"):
            deploy.DeployConfig(checkpoint_path="unused", action_mode="random")


class SafetyContractTest(unittest.TestCase):
    def test_loader_rejects_incomplete_weights(self):
        controller = object.__new__(deploy.GrSimVisionController)
        controller.config = deploy.DeployConfig(checkpoint_path="/tmp/checkpoint")
        controller.device = torch.device("cpu")
        fake_state = {"weights": {}}
        with mock.patch.object(Path, "exists", return_value=True), mock.patch(
            "builtins.open", mock.mock_open(read_data=b"checkpoint")
        ), mock.patch.object(deploy.pickle, "load", return_value=fake_state):
            with self.assertRaises(RuntimeError):
                controller._load_model()

    def test_watchdog_requires_all_entities_fresh(self):
        controller = object.__new__(deploy.GrSimVisionController)
        controller.config = deploy.DeployConfig(checkpoint_path="unused")
        now = 100.0
        controller.entity_updated_at = {
            ("ball", 0): now,
            **{("blue", i): now for i in range(3)},
            **{("yellow", i): now for i in range(3)},
        }
        with mock.patch.object(deploy.time, "monotonic", return_value=now + 0.1):
            self.assertTrue(controller._vision_is_fresh())
        del controller.entity_updated_at[("yellow", 2)]
        with mock.patch.object(deploy.time, "monotonic", return_value=now + 0.1):
            self.assertFalse(controller._vision_is_fresh())

    def test_continuous_mode_resets_temporal_state_only_once(self):
        controller = object.__new__(deploy.GrSimVisionController)
        controller.config = deploy.DeployConfig(checkpoint_path="unused")
        controller.stop_event = mock.Mock()
        controller.stop_event.is_set.side_effect = [False, False, True]
        controller.run_episode = mock.Mock()
        controller._send_zero_commands = mock.Mock()
        controller.vision_socket = None
        controller.command_socket = mock.Mock()

        controller.run_continuous()

        self.assertEqual(controller.run_episode.call_count, 2)
        controller.run_episode.assert_has_calls([
            mock.call(max_steps=None, reset_state=True),
            mock.call(max_steps=None, reset_state=False),
        ])

    def test_kickoff_master_sends_replacement_each_episode(self):
        controller = object.__new__(deploy.GrSimVisionController)
        controller.config = deploy.DeployConfig(
            checkpoint_path="unused", episode_reset=True, kickoff_master=True
        )
        controller.stop_event = mock.Mock()
        controller.stop_event.is_set.side_effect = [False, False, True]
        controller.run_episode = mock.Mock()
        controller.reset_episode = mock.Mock()
        controller._send_zero_commands = mock.Mock()
        controller.vision_socket = None
        controller.command_socket = mock.Mock()

        with mock.patch.object(deploy, "perform_kickoff") as kickoff:
            controller.run_continuous()

        self.assertEqual(kickoff.call_count, 2)
        self.assertEqual(controller.reset_episode.call_count, 2)
        controller.run_episode.assert_has_calls([
            mock.call(max_steps=None, reset_state=False),
            mock.call(max_steps=None, reset_state=False),
        ])

    def test_kickoff_formation_detection_is_edge_triggered(self):
        formation = {
            "ball": [0.02, -0.03],
            "robots_blue": {
                f"robot_{i}": [x, y, d]
                for i, (x, y, d) in enumerate(deploy.KICKOFF_BLUE)
            },
            "robots_yellow": {
                f"robot_{i}": [x, y, d]
                for i, (x, y, d) in enumerate(deploy.KICKOFF_YELLOW)
            },
        }
        self.assertTrue(deploy.is_kickoff_formation(formation))

        moved = {
            **formation,
            "ball": [1.0, 0.5],
        }
        self.assertFalse(deploy.is_kickoff_formation(moved))

        missing_robot = {
            **formation,
            "robots_yellow": {
                key: value
                for key, value in formation["robots_yellow"].items()
                if key != "robot_2"
            },
        }
        self.assertFalse(deploy.is_kickoff_formation(missing_robot))


if __name__ == "__main__":
    unittest.main()