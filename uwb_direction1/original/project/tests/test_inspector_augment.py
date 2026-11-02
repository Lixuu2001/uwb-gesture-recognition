import types
import unittest

import torch

from uwb_osr.inspector import _make_mixup_negatives


class InspectorAugmentTest(unittest.TestCase):
    def test_mixup_negatives_use_different_known_gesture_labels(self):
        cfg = types.SimpleNamespace(
            n_neg_syn=8,
            ood_mix_low=0.5,
            ood_mix_high=0.5,
        )
        features = torch.tensor([[0.0], [10.0]])
        labels = torch.tensor([0, 1])

        synth = _make_mixup_negatives(features, labels, cfg)

        self.assertEqual(tuple(synth.shape), (cfg.n_neg_syn, 1))
        self.assertTrue(torch.allclose(synth, torch.full_like(synth, 5.0)))


if __name__ == "__main__":
    unittest.main()
