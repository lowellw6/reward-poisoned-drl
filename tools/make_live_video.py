"""
Run Atari Pong trajectory from loaded DQN agent and convert to video.

Shows action agent takes at each step and (optionally) specific
q-values (e.g. for target actions) throughout the feed on the 
right-hand side.
"""

import torch
import numpy as np
import cv2
import imageio
import os.path as osp

from rlpyt.envs.atari.atari_env import AtariEnv
from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent

from reward_poisoned_drl.utils import PONG_ACT_MAP

# models poisoned from step 0 on
# MODEL_PREFIX = "/home/lowell/rlpyt/data/local/20210430"
# MODEL_IMM_Dp25 = "054655/rp_fixed_attack/run_0/params.pkl"
# MODEL_IMM_Dp50 = "054748/rp_fixed_attack/run_1/params.pkl"
# MODEL_IMM_D1p0 = "054844/rp_fixed_attack/run_2/params.pkl"
# MODEL_IMM_D2p0 = "054911/rp_fixed_attack/run_3/params.pkl"
# MODEL_FILES = (
#     MODEL_IMM_Dp25,
#     MODEL_IMM_Dp50,
#     MODEL_IMM_D1p0,
#     MODEL_IMM_D2p0
# )

# models poisoned from step 10M on
MODEL_PREFIX = "/home/lowell/reward-poisoned-drl/runs/fixed-attack-first-poison-10M"
MODEL_DEL_Dp25 = "121943/rp_fixed_attack/run_4/params.pkl"
MODEL_DEL_Dp50 = "122002/rp_fixed_attack/run_5/params.pkl"
MODEL_DEL_D1p0 = "122032/rp_fixed_attack/run_6/params.pkl"
MODEL_DEL_D2p0 = "122056/rp_fixed_attack/run_7/params.pkl"
MODEL_FILES = (
    MODEL_DEL_Dp25,
    MODEL_DEL_Dp50,
    MODEL_DEL_D1p0,
    MODEL_DEL_D2p0
)

VID_PREFIX = "/home/lowell/reward-poisoned-drl/data/attack_demos/delay_poison/"

Q_REFS = (  # (q_idx, norm_y_loc)
    (3, 0.85),  # targ_bottom
    (5, 0.6)  # targ_mid
)


def build_frame(step_idx, obs, act, q, q_refs=None):
    """
    Annotate one frame for the live demo,
    using the latest frame in the stack.
    ref_q can be any q-value, and does not 
    necessarily correspond to act.
    """
    scale_factor = 6
    H, W = obs.shape[1:]
    image = cv2.resize(obs[-1], (scale_factor * W, scale_factor * H))
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # using RGB not BGR!
    image = cv2.putText(image,  # frame index
        f"{step_idx}", 
        (5, scale_factor*H-5), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (0, 0, 255),  # R, G, B
        1
    )
    image = cv2.putText(image,  # human-readable action
        PONG_ACT_MAP[act.item()], 
        (int(scale_factor*W / 2) - 20, scale_factor*H-5), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (255, 0, 0),  # R, G, B
        1
    )

    if q_refs is not None:
        fill_width = 128
        fill = np.zeros((scale_factor*H, fill_width, 3), dtype=np.uint8)
        image = np.concatenate((image, fill), axis=1)

        for q_idx, norm_y_loc in q_refs:
            q_val = q[q_idx].item()
            image = cv2.putText(image,
                f"{abs(q_val):6.2f}",
                (scale_factor*W+fill_width-150, int(scale_factor*H*norm_y_loc)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (255, 0, 0) if q_val < 0 else (0, 255, 0),  # R, G, B
                1
            )

    return image


def make_live_video(model_path, q_refs, vid_idx, cuda_idx):
    # create Pong environment
    env = AtariEnv(game="pong")

    # create Pong agent
    agent = AtariDqnAgent()
    agent.initialize(env.spaces)
    agent.to_device(cuda_idx)
    
    # load pre-trained DQN
    agent_state_dict = torch.load(model_path)["agent_state_dict"]["model"]
    agent.load_state_dict(agent_state_dict)

    # dummy values for agent call
    dummy_act = torch.zeros(1)
    dummy_rew = torch.zeros(1)

    # check valid q_refs, if not None
    if q_refs is not None:
        for q_idx, norm_y_loc in q_refs:
            assert q_idx >= 0 and q_idx < env.spaces.action.n
            assert norm_y_loc >= 0 and norm_y_loc <= 1

    # start trajectory, storing annotated frames
    frames = []
    step_idx = 0
    obs = env.reset()
    done = False
    
    while not done:
        q = agent(torch.as_tensor(obs), dummy_act, dummy_rew).detach().numpy()
        act = np.argmax(q)

        next_obs, rew, done, _ = env.step(act)

        ref_q = q[q_idx].item() if q_idx is not None else None

        f = build_frame(step_idx, obs, act, q, q_refs)
        frames.append(f)

        obs = next_obs
        step_idx += 1

    frame_stack = np.stack(frames, axis=0)

    # save both slow and normal speed video
    imageio.mimwrite(osp.join(VID_PREFIX, str(vid_idx) + "_slow.mp4"), frame_stack, fps=3)
    imageio.mimwrite(osp.join(VID_PREFIX, str(vid_idx) + "_norm.mp4"), frame_stack, fps=25)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--cuda_idx', help='gpu to use', type=int, default=None)
    args = parser.parse_args()

    for vid_idx, model_file in enumerate(MODEL_FILES):
        make_live_video(osp.join(MODEL_PREFIX, model_file), Q_REFS, vid_idx, args.cuda_idx)
        print(f"Model {model_file} --> Video {vid_idx}")
