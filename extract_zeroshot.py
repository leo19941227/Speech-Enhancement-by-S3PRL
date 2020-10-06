import os
import re
import sys
from glob import glob
import numpy as np
import torch
import pandas as pd

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

NOISE_TYPE_NUM = 33

def tabulate_events(pattern):

    event_dirs = glob(pattern)
    print(len(event_dirs))

    metrics = ['pesq_nb', 'sisdr', 'stoi']
    table = torch.zeros(NOISE_TYPE_NUM, len(metrics))
    for dname in event_dirs:
        print(f"Converting run {dname}",end="")
        
        # find noise type
        result = re.search('noise(\d+)', dname)
        if result is None:
            continue
        row = int(result.group()[len('noise'):]) - 1

        ea = EventAccumulator(dname).Reload()
        tags = ea.Tags()['scalars']

        for column, tag in enumerate(metrics):
            event = ea.Scalars(f'test_{tag}')[0]
            table[row, column] = event.value

    df = pd.DataFrame(data=table.numpy(), columns=metrics, index=[f'noise{i+1}' for i in range(NOISE_TYPE_NUM)])
    return df

if __name__ == '__main__':
    df = tabulate_events(sys.argv[1])
    df.to_csv(sys.argv[2])
