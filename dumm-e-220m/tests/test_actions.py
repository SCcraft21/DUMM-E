"""Unit tests for safe local action execution and Conscience clearance gating."""

import unittest

from src.voice.actions import SafeCalculator, LocalActionDispatcher


class TestActionsAndConscienceGating(unittest.TestCase):
    """Test suite for local action dispatcher and moral clearance protocol."""

    def setUp(self) -> None:
        self.dispatcher = LocalActionDispatcher(default_risk_threshold=0.50)

    def test_safe_calculator(self) -> None:
        """Verify arithmetic evaluator computes expressions accurately."""
        self.assertEqual(SafeCalculator.evaluate("12 + 28"), 40)
        self.assertEqual(SafeCalculator.evaluate("(100 - 25) * 4"), 300)
        self.assertEqual(SafeCalculator.evaluate("2 ** 8"), 256)

        # Confirm arbitrary code execution attempt is blocked by parser
        with self.assertRaises(ValueError):
            SafeCalculator.evaluate("__import__('os').system('ls')")

    def test_conscience_clearance_approved(self) -> None:
        """Verify that safe actions with low conscience risk are executed."""
        action_payload = {"action": "calculator", "params": {"expression": "25 * 4"}}
        # Safe risk score (0.10 <= 0.50 threshold)
        obs = self.dispatcher.execute_action(
            action_payload=action_payload,
            conscience_risk_score=0.10,
        )
        self.assertEqual(obs["status"], "success")
        self.assertEqual(obs["clearance"], "APPROVED")
        self.assertEqual(obs["result"]["result"], 100)

    def test_conscience_clearance_denied(self) -> None:
        """Verify that actions with high conscience risk are blocked before execution."""
        action_payload = {"action": "calculator", "params": {"expression": "25 * 4"}}
        # High risk score (0.85 > 0.50 threshold)
        obs = self.dispatcher.execute_action(
            action_payload=action_payload,
            conscience_risk_score=0.85,
        )
        self.assertEqual(obs["status"], "blocked")
        self.assertEqual(obs["clearance"], "DENIED")
        self.assertIn("blocked", obs["error"])

    def test_system_info_action(self) -> None:
        """Verify system info tool returns structured telemetry."""
        action_payload = {"action": "system_info", "params": {"metric": "os"}}
        obs = self.dispatcher.execute_action(
            action_payload=action_payload,
            conscience_risk_score=0.05,
        )
        self.assertEqual(obs["status"], "success")
        self.assertIn("os", obs["result"])


if __name__ == "__main__":
    unittest.main()

