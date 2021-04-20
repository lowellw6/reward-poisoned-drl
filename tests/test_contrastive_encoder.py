
import unittest
import numpy as np
import cv2
import torch
import pickle
from tqdm import tqdm

from reward_poisoned_drl.contrastive_encoder.data_generator import ContrastiveDG, DummyContrastiveDG
from reward_poisoned_drl.utils import show_frame_stacks, random_crop, semantic_crop_pong
from reward_poisoned_drl.contrastive_encoder.encoder import PixelEncoder

USE_DUMMY = False


def get_single_ob():
    ob_path = "/home/lowell/reward-poisoned-drl/data/single_ob.pkl"
    with open(ob_path, "rb") as f:
        ob = pickle.load(f)
    return ob


# class TestContrastiveDG(unittest.TestCase):

#     gen = None

#     @classmethod
#     def setUpClass(cls):
#         dataset_dir = "/home/lowell/reward-poisoned-drl/data"
#         GenCls = DummyContrastiveDG if USE_DUMMY else ContrastiveDG
#         cls.gen = GenCls(dataset_dir)  # takes a while if not using dummy class

#     def test_init_loading(self):
#         """
#         Checks each frame was written to other than first trailing
#         stacked frames in initial 1M replay buffer. This test assumes 
#         all data comes from one DqnStore run and includes the first 1M
#         samples taken.
#         """
#         zero_frames = np.sum(np.logical_not(np.any(TestContrastiveDG.gen.frames, axis=(3, 4))))
#         self.assertEqual(zero_frames, (TestContrastiveDG.gen.frame_stack - 1) * TestContrastiveDG.gen.data_shape[2])  # shape[2] == B
    
#     def test_uniform_sampling(self):
#         bs = 8
#         samples = TestContrastiveDG.gen._uniform_sample(bs)
#         H, W = TestContrastiveDG.gen.data_shape[3:]
#         frame_stack = TestContrastiveDG.gen.frame_stack
#         self.assertEqual(samples.shape, (bs, frame_stack, H, W))

#     def test_viz_uniform_samples(self):
#         samples = TestContrastiveDG.gen._uniform_sample(4)
#         show_frame_stacks(samples, "Uniform Samples")

    # def test_generator(self):
    #     """Includes random cropping augmentations."""
    #     gen = TestContrastiveDG.gen

    #     def check_batch(gen, batch_size, batch_shape):
    #         H, W = gen.data_shape[3:]
    #         bs, fs, h, w = batch_shape
    #         self.assertGreater(bs, 0)
    #         self.assertLessEqual(bs, batch_size)
    #         self.assertEqual(fs, gen.frame_stack)
    #         self.assertEqual(h, H - gen.H_reduce)
    #         self.assertEqual(w, W - gen.W_reduce)

    #     def check_gen(gen, batch_size):
    #         print(f"Testing generator --> batch-size {batch_size}")
    #         for anch, pos in tqdm(gen.generator(batch_size)):
    #             check_batch(gen, batch_size, anch.shape)
    #             check_batch(gen, batch_size, pos.shape)
                
    #     check_gen(gen, 64)
    #     check_gen(gen, 1024)

    # def test_viz_generator_samples(self):
    #     """Includes random cropping augmentations."""
    #     anch, pos = next(TestContrastiveDG.gen.generator(4))
    #     show_frame_stacks(anch.to(torch.uint8).numpy(), f"Anchors {anch.shape[2]}x{anch.shape[3]}")
    #     show_frame_stacks(pos.to(torch.uint8).numpy(), f"Positives {pos.shape[2]}x{pos.shape[3]}")


# class TestAugmentation(unittest.TestCase):

    # def test_semantic_crop(self):
    #     """Gets rid of adversary and paddle, which we are not concerned with."""
    #     ob = get_single_ob()
    #     ob = np.expand_dims(ob.transpose(2, 0, 1), axis=0)  # H, W, C --> 1, C, H, W
    #     show_frame_stacks(ob, "Original")
    #     ob = semantic_crop_pong(ob)
    #     self.assertEqual(ob.shape, (1, 4, 84, 70))
    #     show_frame_stacks(ob, "Semantic Crop")
    
    # def test_random_cropping(self):
    #     """Quickly load single frame stack for checking cropping regions."""
    #     H_reduce = 8
    #     W_reduce = 8

    #     ob = get_single_ob()
    #     ob = np.expand_dims(ob.transpose(2, 0, 1), axis=0)  # H, W, C --> 1, C, H, W
    #     ob = semantic_crop_pong(ob)

    #     # get crops in corners to show worst-case omissions
    #     upper_left = ob[:, :, :-H_reduce, :-W_reduce]
    #     upper_right = ob[:, :, :-H_reduce, W_reduce:]
    #     bottom_right = ob[:, :, H_reduce:,  W_reduce:]
    #     bottom_left = ob[:, :, H_reduce:, :-W_reduce]

    #     show_frame_stacks(ob, "Original")
    #     cat = np.concatenate((upper_left, upper_right, bottom_right, bottom_left), axis=0)
    #     show_frame_stacks(cat, "Extreme Crops (Clockwise)")


class TestPixelEncoder(unittest.TestCase):

    ### UNNECESSARY - Just modifying PixelEncoder details slightly to match shape
    # def test_resize(self):
    #     ob = get_single_ob()
    #     H, W = ob.shape[:2]
    #     ob = ob[4:H-4, 4:W-4, :]  # center crop with reduce = 8
    #     orig = ob.copy()
    #     resized = cv2.resize(ob, (84, 84))  # to match CURL
    #     show_frame_stacks(np.expand_dims(orig.transpose(2, 0, 1), axis=0), "Center Crop")
    #     show_frame_stacks(np.expand_dims(resized.transpose(2, 0, 1), axis=0), "Resized Crop")
    
    def test_basics(self):
        ob = get_single_ob()
        ob = np.expand_dims(ob.transpose(2, 0, 1), axis=0)  # H, W, C --> 1, C, H, W
        ob = semantic_crop_pong(ob)
        H, W = ob.shape[2:]
        ob = ob[:, :, 4:H-4, 4:W-4]  # center crop with reduce = 8
        
        self.assertEqual(ob.shape, (1, 4, 76, 62))
        
        ob = torch.as_tensor(ob, dtype=torch.float32)
        
        enc = PixelEncoder()
        out = enc(ob)

        self.assertEqual(out.shape, (1, 128))


if __name__ == "__main__":
    unittest.main()