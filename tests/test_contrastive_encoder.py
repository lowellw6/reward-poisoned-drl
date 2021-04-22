
import unittest
import numpy as np
import cv2
import torch
import pickle
from tqdm import tqdm
import copy

from reward_poisoned_drl.contrastive_encoder.data_generator import ContrastiveDG, DummyContrastiveDG
from reward_poisoned_drl.utils import soft_update_params, show_frame_stacks, random_crop, semantic_crop_pong
from reward_poisoned_drl.contrastive_encoder.encoder import PixelEncoder
from reward_poisoned_drl.contrastive_encoder.contrast import ContrastiveTrainer

USE_DUMMY = False


def get_single_ob():
    ob_path = "/home/lowell/reward-poisoned-drl/data/single_ob.pkl"
    with open(ob_path, "rb") as f:
        ob = pickle.load(f)
    return ob


def get_single_input():
    """Need a bit of prep for contrastive encoder models."""
    ob = get_single_ob()
    ob = np.expand_dims(ob.transpose(2, 0, 1), axis=0)  # H, W, C --> 1, C, H, W
    ob = semantic_crop_pong(ob)
    H, W = ob.shape[2:]
    ob = ob[:, :, 4:H-4, 4:W-4]  # center crop with reduce = 8        
    return torch.as_tensor(ob, dtype=torch.float32)


# class TestContrastiveDG(unittest.TestCase):

#     gen = None

#     @classmethod
#     def setUpClass(cls):
#         dataset_dir = "/home/lowell/reward-poisoned-drl/data/train"
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


class TestAugmentation(unittest.TestCase):

    def test_semantic_crop(self):
        """Gets rid of adversary and paddle, which we are not concerned with."""
        ob = get_single_ob()
        ob = np.expand_dims(ob.transpose(2, 0, 1), axis=0)  # H, W, C --> 1, C, H, W
        show_frame_stacks(ob, "Original")
        ob = semantic_crop_pong(ob)
        self.assertEqual(ob.shape, (1, 4, 84, 70))
        show_frame_stacks(ob, "Semantic Crop")
    
    def test_random_cropping(self):
        """Quickly load single frame stack for checking cropping regions."""
        H_reduce = 8
        W_reduce = 8

        ob = get_single_ob()
        ob = np.expand_dims(ob.transpose(2, 0, 1), axis=0)  # H, W, C --> 1, C, H, W
        ob = semantic_crop_pong(ob)

        # get crops in corners to show worst-case omissions
        upper_left = ob[:, :, :-H_reduce, :-W_reduce]
        upper_right = ob[:, :, :-H_reduce, W_reduce:]
        bottom_right = ob[:, :, H_reduce:,  W_reduce:]
        bottom_left = ob[:, :, H_reduce:, :-W_reduce]

        show_frame_stacks(ob, "Original")
        cat = np.concatenate((upper_left, upper_right, bottom_right, bottom_left), axis=0)
        show_frame_stacks(cat, "Extreme Crops (Clockwise)")


# class TestPixelEncoder(unittest.TestCase):

#     ### UNNECESSARY - Just modifying PixelEncoder details slightly to match shape
#     # def test_resize(self):
#     #     ob = get_single_ob()
#     #     H, W = ob.shape[:2]
#     #     ob = ob[4:H-4, 4:W-4, :]  # center crop with reduce = 8
#     #     orig = ob.copy()
#     #     resized = cv2.resize(ob, (84, 84))  # to match CURL
#     #     show_frame_stacks(np.expand_dims(orig.transpose(2, 0, 1), axis=0), "Center Crop")
#     #     show_frame_stacks(np.expand_dims(resized.transpose(2, 0, 1), axis=0), "Resized Crop")
    
#     def test_basics(self):
#         ob = get_single_ob()
#         ob = np.expand_dims(ob.transpose(2, 0, 1), axis=0)  # H, W, C --> 1, C, H, W
#         ob = semantic_crop_pong(ob)
#         H, W = ob.shape[2:]
#         ob = ob[:, :, 4:H-4, 4:W-4]  # center crop with reduce = 8
        
#         self.assertEqual(ob.shape, (1, 4, 76, 62))
        
#         ob = torch.as_tensor(ob, dtype=torch.float32)
        
#         enc = PixelEncoder()
#         out = enc(ob)

#         self.assertEqual(out.shape, (1, 50))


# class TestContrastiveTrainer(unittest.TestCase):

#     def setUp(self):
#         self.device = torch.device("cuda:0")
#         self.trainer = ContrastiveTrainer(self.device)

#     def test_compute_logits(self):
#         trainer = self.trainer
        
#         anch = get_single_input().to(self.device)
#         pos = anch.clone()
 
#         z_a = trainer.query_enc(anch)
#         z_pos = trainer.key_enc(pos).detach()
#         logits = trainer.compute_logits(z_a, z_pos)

#         self.assertTrue(logits.item() == 0.)

#     def test_gradient_back_prop(self):
#         trainer = self.trainer

#         anch = get_single_input().to(self.device)
#         pos = anch.clone()
 
#         z_a = trainer.query_enc(anch)
#         z_pos = trainer.key_enc(pos).detach()

#         logits = trainer.compute_logits(z_a, z_pos)
#         labels = torch.arange(logits.shape[0]).long().to(self.device)
#         loss = trainer.cross_entropy_loss(logits, labels)

#         for param in trainer.query_enc.parameters():
#             self.assertIsNone(param.grad)

#         for param in trainer.key_enc.parameters():
#             self.assertIsNone(param.grad)

#         self.assertIsNone(trainer.W.grad)
        
#         trainer.opt.zero_grad()
#         loss.backward()

#         for param in trainer.query_enc.parameters():
#             self.assertIsNotNone(param.grad)
        
#         for param in trainer.key_enc.parameters():
#             self.assertIsNone(param.grad)  # is detached

#         self.assertIsNotNone(trainer.W.grad)

#     def test_momentum_target_update(self):
#         trainer = self.trainer

#         query_enc_params_before = copy.deepcopy(list(trainer.query_enc.parameters()))
#         key_enc_params_before = copy.deepcopy(list(trainer.key_enc.parameters()))
        
#         soft_update_params(trainer.query_enc, trainer.key_enc, 0.5)  # large tau to exaggerate average

#         for p1, p2 in zip(trainer.query_enc.parameters(), query_enc_params_before):
#             self.assertTrue(torch.allclose(p1, p2))

#         for p1, p2 in zip(trainer.key_enc.parameters(), key_enc_params_before):
#             one_or_zeros = torch.prod(p1) == 0. or torch.prod(p1) == 1.
#             self.assertTrue(one_or_zeros or not torch.allclose(p1, p2))


if __name__ == "__main__":
    unittest.main()