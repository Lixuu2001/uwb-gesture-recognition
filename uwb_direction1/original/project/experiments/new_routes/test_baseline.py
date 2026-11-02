import unittest
import numpy as np
import torch

from train_baseline import T, extract_batch, source_parts


class BaselineContractTest(unittest.TestCase):
    def test_whole_users_and_quality_mask(self):
        _, y, subjects, parts = source_parts(1)
        self.assertEqual(sum(map(len, parts.values())), 9300)
        self.assertEqual(set(subjects[parts['train']]), {3, 4, 5, 6, 7, 8})
        self.assertEqual(set(subjects[parts['val']]), {2})
        self.assertEqual(set(subjects[parts['test']]), {1})
        for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
            self.assertFalse(set(parts[a]) & set(parts[b]))
        self.assertFalse(np.any((subjects[parts['train']] == 6) & (y[parts['train']] == 3)))

    def test_extraction_preserves_classifier(self):
        torch.set_num_threads(2)
        model = T.models.Classifier([2], 'concat', 'mean').eval()
        x = torch.randn(2, 1, 90, 378)
        with torch.no_grad():
            expected = model(x)[0]
            features, actual = extract_batch(model, x)
            rebuilt = model.head.net[-1](features)
        self.assertEqual(tuple(features.shape), (2, 256))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual, rebuilt, rtol=0, atol=0)
        self.assertFalse(model.head.net[4]._forward_hooks)


if __name__ == '__main__':
    unittest.main()
