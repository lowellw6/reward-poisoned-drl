"""
Shared project utilities.
Adapted from: https://github.com/MishaLaskin/curl/blob/master/utils.py
"""


import torch
import numpy as np
from skimage.util.shape import view_as_windows
import cv2

# Pong pre-processes cropping hyperparam constants
PONG_CROP = dict(
    top=14,
    bottom=6,
    left=10
)
# no right crop


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def random_crop(imgs, output_size):
    """
    Vectorized way to do random crop using sliding windows
    and picking out random ones
    args:
        imgs, batch images with shape (B,C,H,W)
    """
    # batch size
    n = imgs.shape[0]
    H, W = imgs.shape[2:]
    H_out, W_out = output_size
    h_crop_max = H - H_out
    w_crop_max = W - W_out
    imgs = np.transpose(imgs, (0, 2, 3, 1))
    w1 = np.random.randint(0, w_crop_max, n)
    h1 = np.random.randint(0, h_crop_max, n)
    # creates all sliding windows combinations of size (output_size)
    windows = view_as_windows(
        imgs, (1, H_out, W_out, 1))[..., 0,:,:, 0]
    # selects a random window for each batch element
    cropped_imgs = windows[np.arange(n), w1, h1]
    return cropped_imgs


def center_crop_image(image, output_size):
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    image = image[:, top:top + new_h, left:left + new_w]
    return image


def semantic_crop_pong(obs):
    """
    Removes adversary paddle and score in pong,
    as we only care about the position of the agent
    paddle and the puck for targeted attacks.
    Image-stack shape: (?, 4, 104, 80) --> (?, 4, 84, 70)
    """
    H, W = obs.shape[2:]
    assert (H, W) == (104, 80)
    t, b, l = PONG_CROP["top"], PONG_CROP["bottom"], PONG_CROP["left"]
    return obs[:, :, t:H-b, l:]


def show_frame_stacks(batch, window_name):
    H, W = batch.shape[2:]
    scale_factor = 6
    for bs_idx in range(len(batch)):
        for fs_idx in range(batch.shape[1]):
            cv2.waitKey(500)
            image = cv2.resize(batch[bs_idx, fs_idx, :, :], (scale_factor * W, scale_factor * H))
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            image = cv2.putText(image, 
                f"{bs_idx}-{fs_idx}", 
                (5, scale_factor*H-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 0, 255),  # B, G, R 
                1
            )
            cv2.imshow(window_name, image)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
