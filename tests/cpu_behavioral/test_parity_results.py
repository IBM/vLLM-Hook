"""Comparison regression checks; no vLLM model or GPU required."""
import copy
import math
import unittest

from parity_results import compare_results, compare_values


class ComparisonTests(unittest.TestCase):
    def compare(self, a, b):
        return compare_values(a, b, atol=1e-4, rtol=1e-3)

    def test_tolerance_and_report(self):
        result = self.compare({'tensor': [1.0, 2.0]}, {'tensor': [1.0001, 2.0002]})
        self.assertTrue(result['passed'])
        self.assertEqual(result['numbers_compared'], 2)
        self.assertGreater(result['max_absolute_error'], 0)

    def test_small_relative_increase_accepts_observed_steering_difference(self):
        cpu, cuda = [-0.16367053985595703], [-0.16340255737304688]
        self.assertFalse(compare_values(cpu, cuda, atol=1e-4, rtol=1e-3)['passed'])
        self.assertTrue(compare_values(cpu, cuda, atol=1e-4, rtol=1.1e-3)['passed'])
        self.assertFalse(compare_values(cpu, [-0.16], atol=1e-4, rtol=1.1e-3)['passed'])

    def test_detects_wrong_values(self):
        self.assertFalse(self.compare([1.0], [2.0])['passed'])

    def test_detects_shape_and_missing_layer(self):
        self.assertFalse(self.compare({'layer1': [1.0]}, {'layer2': [1.0]})['passed'])
        self.assertFalse(self.compare([[1.0]], [1.0])['passed'])

    def test_ranking_is_exact(self):
        self.assertFalse(self.compare({'scores': [1.0], 'rank': [0, 1]},
                                      {'scores': [1.0], 'rank': [1, 0]})['passed'])

    def test_nonfinite_is_not_a_pass(self):
        for value in (math.nan, math.inf, -math.inf):
            self.assertFalse(self.compare([value], [value])['passed'])

    def test_empty_comparison_is_not_a_pass(self):
        self.assertFalse(self.compare([], [])['passed'])
        self.assertFalse(self.compare({'shape': [1]}, {'shape': [1]})['passed'])

    def test_mismatched_inputs_and_wrong_backend_fail(self):
        cpu = {'backend': 'cpu', 'metadata': {'model_revision': 'abc'}, 'values': [1.0]}
        cuda = copy.deepcopy(cpu)
        cuda['backend'] = 'cuda'
        self.assertTrue(compare_results(cpu, cuda, atol=1e-4, rtol=1e-3)['passed'])
        cuda['metadata']['model_revision'] = 'different'
        with self.assertRaises(ValueError):
            compare_results(cpu, cuda, atol=1e-4, rtol=1e-3)
        cuda['backend'] = 'cpu'
        with self.assertRaises(ValueError):
            compare_results(cpu, cuda, atol=1e-4, rtol=1e-3)

    def test_invalid_tolerances(self):
        for tolerance in (-1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                compare_values([1.0], [1.0], atol=tolerance, rtol=0)


if __name__ == '__main__':
    unittest.main()
