
import unittest
import numpy as np
import pickle
import os


class TestDQNStore(unittest.TestCase):
    
    def test_large_file_dump(self):
        """
        To circumvent Pickle bug which occurs
        when writing files larger than 4 GB.
        """
        dummy_buffer = dict(
            observation=np.zeros((125000, 8, 104, 80), dtype=np.uint8),
            action=np.zeros((125000, 8), dtype=np.int64),
            reward=np.zeros((125000, 8), dtype=np.float32),
            done=np.zeros((125000, 8), dtype=np.bool)
        )
        dump_name = "temp.pkl"
        mem_size = sum([x.size + x.itemsize for x in dummy_buffer.values()])
        max_chunk_size = int(4e9)  # 4 GB
        if mem_size > max_chunk_size:
            num_splits = (mem_size // max_chunk_size) + 1
            chunks = {key: np.array_split(dummy_buffer[key], num_splits, axis=0) for key in dummy_buffer.keys()}
            for idx in range(num_splits):
                chunk = {key: val[idx] for key, val in chunks.items()}
                with open(f"temp{idx}.pkl", "wb") as save_file:
                    pickle.dump(chunk, save_file)


if __name__ == "__main__":
    unittest.main()