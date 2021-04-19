
import unittest
import numpy as np

from reward_poisoned_drl.contrastive_encoder.data_generator import ContrastiveDG


class TestContrastiveDG(unittest.TestCase):

    def test_init_loading(self):
        """
        Checks each frame was written to other than first trailing
        stacked frames in initial 1M replay buffer. This test assumes 
        all data comes from one DqnStore run and includes the first 1M
        samples taken.
        """
        dataset_dir = "/home/lowell/reward-poisoned-drl/data"
        gen = ContrastiveDG(dataset_dir)
        zero_frames = np.sum(np.logical_not(np.any(gen.frames, axis=(3, 4))))
        self.assertEqual(zero_frames, (gen.frame_stack - 1) * gen.B)


class TestPixelEncoder(unittest.TestCase):
    
    def test_TODO(self):
        pass


if __name__ == "__main__":
    unittest.main()