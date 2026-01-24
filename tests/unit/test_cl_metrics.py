import unittest
from tensorguard.metrics.continual_learning import compute_cl_metrics

class TestCLMetrics(unittest.TestCase):
    def test_basic_metrics(self):
        # 2x2 matrix
        # Task 0: 0.8 -> 0.7 (forgetting 0.1)
        # Task 1: 0.0 -> 0.9
        eval_matrix = [
            [0.8, 0.0],
            [0.7, 0.9]
        ]
        
        results = compute_cl_metrics(eval_matrix)
        
        # ACC = (0.7 + 0.9) / 2 = 0.8
        self.assertAlmostEqual(results["avg_accuracy"], 0.8)
        
        # Forgetting for Task 0 = 0.8 - 0.7 = 0.1
        self.assertAlmostEqual(results["forgetting_mean"], 0.1)
        
        # BWT = (0.7 - 0.8) = -0.1
        self.assertAlmostEqual(results["bwt"], -0.1)

    def test_single_task(self):
        eval_matrix = [[0.9]]
        results = compute_cl_metrics(eval_matrix)
        self.assertEqual(results["avg_accuracy"], 0.9)
        self.assertEqual(results["forgetting_mean"], 0.0)
        self.assertEqual(results["bwt"], 0.0)

if __name__ == "__main__":
    unittest.main()
