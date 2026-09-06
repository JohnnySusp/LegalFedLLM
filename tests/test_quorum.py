from __future__ import annotations

import unittest

from coordinator.quorum import TrustedClientQuorumPolicy


class TrustedClientQuorumPolicyTests(unittest.TestCase):
    def test_majority_with_a_minimum_of_two(self) -> None:
        policy = TrustedClientQuorumPolicy(minimum=2)

        self.assertEqual(policy.resolve(2), 2)
        self.assertEqual(policy.resolve(3), 2)
        self.assertEqual(policy.resolve(4), 3)
        self.assertEqual(policy.resolve(5), 3)

    def test_one_client_requires_an_explicit_temporary_override(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "below the minimum trusted Client quorum",
        ):
            TrustedClientQuorumPolicy(minimum=2).resolve(1)

        self.assertEqual(
            TrustedClientQuorumPolicy(minimum=2, override=1).resolve(1),
            1,
        )

    def test_override_cannot_exceed_selected_client_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "override exceeds"):
            TrustedClientQuorumPolicy(minimum=2, override=2).resolve(1)

    def test_invalid_policy_values_fail_closed(self) -> None:
        for kwargs in ({"minimum": 0}, {"override": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TrustedClientQuorumPolicy(**kwargs)

        for count in (0, -1, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                TrustedClientQuorumPolicy().resolve(count)


if __name__ == "__main__":
    unittest.main()
