
import numpy as np
import imageio
import pickle
import os.path as osp
import os


def make_video(path):
    out_dir = osp.join(path, "videos")
    if not osp.exists(out_dir):
        os.mkdir(out_dir)

    files = filter(lambda x: x.endswith(".pkl"), os.listdir(path))
    for file_name in files:
        data_path = osp.join(path, file_name)
        base_name = "".join(file_name.split('.')[:-1])

        with open(data_path, "rb") as read_file:
            buffer_dict = pickle.load(read_file)
            print(buffer_dict["observation"].shape)
            obs = buffer_dict["observation"]
            T, B, H, W = obs.shape
            for b in range(B):
                b_obs = obs[:, b, :, :]
                frame_list = [frame.squeeze() for frame in np.split(b_obs, T, axis=0)]
                out_name = osp.join(out_dir, f"{base_name}_B{b}.mp4")
                imageio.mimwrite(out_name, frame_list, fps=25)
                del frame_list


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', help='Path to pickled replay buffer')
    args = parser.parse_args()
    make_video(args.path)