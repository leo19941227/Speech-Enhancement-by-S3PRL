import os
import sys
import random
from librosa.util import find_files
from ipdb import set_trace

SAMPLE_NUM = 10

files = find_files(sys.argv[1])
files = sorted(files)

random.seed(1227)
random.shuffle(files)

pattern = 'LibriSpeech/'
new_files = []
for pth in files:
    start = pth.find(pattern) + len(pattern)
    new_files.append(pth[start:])
files = new_files

with open('libri-dev-all.txt', 'w') as handle:
    for line in files:
        handle.write(f'{line}\n')

adapt = random.sample(files, SAMPLE_NUM)
with open('libri-dev-adapt.txt', 'w') as handle:
    for line in adapt:
        handle.write(f'{line}\n')
