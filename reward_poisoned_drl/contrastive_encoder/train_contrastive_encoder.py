"""
Trains observation-encoder using self-supervised contrastive
objective on Atari Pong. Here we use instance discrimination 
with random crop augmentation, following hyperparameters
of the contrastive objective found in CURL:
https://github.com/MishaLaskin/curl
"""

import torch
import os.path as osp
from tqdm import tqdm

from reward_poisoned_drl.contrastive_encoder.data_generator import ContrastiveDG, DummyContrastiveDG
from reward_poisoned_drl.contrastive_encoder.contrast import ContrastiveTrainer
from reward_poisoned_drl.utils import basic_stats
from reward_poisoned_drl.logger import LoggerLight

TRAIN_DATASET_DIR = "/home/lowell/reward-poisoned-drl/data/train"
VAL_DATASET_DIR = "/home/lowell/reward-poisoned-drl/data/val"

LOG_DIR = "/home/lowell/reward-poisoned-drl/runs/contrast_enc_4_20"
MODEL_LOAD_PATH = None


def build_and_train(cuda_idx=None, epochs=50, batch_size=32, max_val_batch=100, debug=False):
    device = torch.device("cpu") if cuda_idx is None else torch.device(cuda_idx)
    DgCls = DummyContrastiveDG if debug else ContrastiveDG

    train_loader = DgCls(TRAIN_DATASET_DIR, device=device)
    val_loader = DgCls(VAL_DATASET_DIR, device=device)
    
    trainer = ContrastiveTrainer(device=device)

    log_keys = ["tloss_min", "tloss_mean", "tloss_max", "tloss_std", 
                "vloss_min", "vloss_mean", "vloss_max", "vloss_std"]
    logger = LoggerLight(LOG_DIR, log_keys)

    print(f"Training for {epochs} epochs at batch-size {batch_size}")

    if MODEL_LOAD_PATH is not None:
        print(f"Loading model --> {MODEL_LOAD_PATH}")
        trainer.load_state_dict(torch.load(MODEL_LOAD_PATH))
    else:
        print("Random weight init")

    print(f"Logging--> {LOG_DIR}")
    print("Loss stat format: min mean max std")

    for e_idx in range(1, epochs + 1):

        print(f"Epoch {e_idx}/{epochs}")

        print("Train...")
        train_losses = []
        trainer.train()
        for t_idx, batch in tqdm(enumerate(train_loader.generator(batch_size))):
            anch, pos = batch
            tl = trainer.update(anch, pos, t_idx)
            train_losses.append(tl)

        print("Eval...")
        trainer.eval()
        val_losses = []
        for v_idx, batch in enumerate(val_loader.generator(batch_size, shuffle=False)):
            if v_idx >= max_val_batch:
                break
            anch, pos = batch
            vl = trainer.test(anch, pos)
            val_losses.append(vl)

        t_min, t_mean, t_max, t_std = basic_stats(train_losses)
        v_min, v_mean, v_max, v_std = basic_stats(val_losses)

        logger.dump([t_min, t_mean, t_max, t_std, v_min, v_mean, v_max, v_std])

        print(f"Train-loss:  {t_min:.2f} {t_mean:.2f} {t_max:.2f} {t_std:.2f}")
        print(f"  Val-loss:  {v_min:.2f} {v_mean:.2f} {v_max:.2f} {v_std:.2f}")

        model_file_name = f"contrast_enc_{e_idx}.pt"
        print(f"Saving {model_file_name}")
        torch.save(trainer.state_dict(), osp.join(LOG_DIR, model_file_name))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--cuda_idx', help='gpu to use', type=int, default=None)
    parser.add_argument('-e', '--epochs', help='number of passes through training set', type=int, default=50)
    parser.add_argument('-b', '--batch_size', help='number of negatives plus one (for positive) for each anchor', type=int, default=32)
    parser.add_argument('-v', '--max_val_batch', help='max number of eval batches to run each epoch', type=int, default=100)
    parser.add_argument('-d', '--debug', action='store_true', help='use dummy data gen to avoid large data load', default=False)
    args = parser.parse_args()
    build_and_train(
        cuda_idx=args.cuda_idx,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_val_batch=args.max_val_batch,
        debug=args.debug
    )
