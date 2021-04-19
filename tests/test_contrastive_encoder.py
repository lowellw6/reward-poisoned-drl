
import unittest
import numpy as np
import cv2

from reward_poisoned_drl.contrastive_encoder.data_generator import ContrastiveDG, DummyDataGenerator

USE_DUMMY = False


class TestContrastiveDG(unittest.TestCase):

    gen = None

    @classmethod
    def setUpClass(cls):
        dataset_dir = "/home/lowell/reward-poisoned-drl/data"
        GenCls = DummyDataGenerator if USE_DUMMY else ContrastiveDG
        cls.gen = ContrastiveDG(dataset_dir)  # takes a while if not using dummy class

    def test_init_loading(self):
        """
        Checks each frame was written to other than first trailing
        stacked frames in initial 1M replay buffer. This test assumes 
        all data comes from one DqnStore run and includes the first 1M
        samples taken.
        """
        zero_frames = np.sum(np.logical_not(np.any(TestContrastiveDG.gen.frames, axis=(3, 4))))
        self.assertEqual(zero_frames, (TestContrastiveDG.gen.frame_stack - 1) * TestContrastiveDG.gen.data_shape[2])  # shape[2] == B
    
    def test_sampling(self):
        bs = 8
        samples = TestContrastiveDG.gen._uniform_sample(bs)
        H, W = TestContrastiveDG.gen.data_shape[3:]
        frame_stack = TestContrastiveDG.gen.frame_stack
        self.assertEqual(samples.shape, (bs, frame_stack, H, W))
        for bs_idx in range(bs):
            for fs_idx in range(frame_stack):
                cv2.waitKey(500)
                image = cv2.resize(samples[bs_idx, fs_idx, :, :], (6 * W, 6 * H))
                image = cv2.putText(image, 
                    f"{bs_idx}-{fs_idx}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (255, 255, 255),  # B, G, R 
                    0
                )
                h, w = image.shape
                cv2.imshow("Sample Frames", image)
            cv2.waitKey(0)
        cv2.destroyAllWindows()


class TestPixelEncoder(unittest.TestCase):
    
    def test_TODO(self):
        pass


if __name__ == "__main__":
    unittest.main()