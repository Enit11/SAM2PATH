import unittest

import numpy as np

from evaluate_segmentation import boundary_f1, dice_iou, evaluate_pair, hd95


class EvaluationSanityTests(unittest.TestCase):
    def test_perfect_prediction(self):
        target = np.zeros((20, 20), dtype=np.uint8)
        target[2:8, 2:8] = 1
        target[10:15, 2:8] = 2
        target[2:8, 10:15] = 3
        result = evaluate_pair(target.copy(), target, "perfect")
        self.assertEqual(result["dice_fg_macro"], 1.0)
        self.assertEqual(result["iou_fg_macro"], 1.0)
        self.assertEqual(result["bf1_1px_fg_macro"], 1.0)
        self.assertEqual(result["hd95_fg_macro"], 0.0)

    def test_dice_iou_identity(self):
        pred = np.zeros((10, 10), dtype=bool); pred[:5, :5] = True
        target = np.zeros((10, 10), dtype=bool); target[2:7, :5] = True
        dice, iou, *_ = dice_iou(pred, target)
        self.assertAlmostEqual(dice, 2 * iou / (1 + iou), places=12)

    def test_empty_convention(self):
        empty = np.zeros((16, 16), dtype=bool)
        self.assertEqual(dice_iou(empty, empty)[:2], (1.0, 1.0))
        self.assertEqual(boundary_f1(empty, empty, 1), 1.0)
        self.assertEqual(hd95(empty, empty), 0.0)


if __name__ == "__main__":
    unittest.main()
