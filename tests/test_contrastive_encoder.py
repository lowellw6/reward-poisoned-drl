
import unittest

from reward_poisoned_rl.contrastive_encoder.data_generator import ContrastDataGenerator


class TestContrastDataGenerator(unittest.TestCase):

    def test_init_loading(self):
        dataset_dir = "~/reward-poisoned-rl/contrastive-encoder/data"
        gen = ContrastDataGenerator(dataset_dir)


class TestPixelEncoder(unittest.TestCase):
    
    def test_TODO(self):
        pass


if __name__ == "__main__":
    unittest.main()