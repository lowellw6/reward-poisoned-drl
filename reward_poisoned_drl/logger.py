"""
Lightweight tabular logger
"""

import csv
import os
import os.path as osp

LOG_FILE = "progress.csv"


class LoggerLight:

    def __init__(self, log_dir, keys):
        if not osp.exists(log_dir):
            os.makedirs(log_dir)

        log_path = osp.join(log_dir, LOG_FILE)

        with open(log_path, 'x', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(keys)

        self.log_path = log_path
        self.keys = keys

    def dump(self, vals):
        assert len(vals) == len(self.keys)
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(vals)
