import os
import random

random.seed(1227)
lines = open('libri-test-clean-10s.txt', 'r').readlines()
random.shuffle(lines)

adapt = lines[:10]
test = lines[10:1210]

with open('libri-adapt.txt', 'w') as handle:
    for line in adapt:
        handle.write(line)

with open('libri-test.txt', 'w') as handle:
    for line in test:
        handle.write(line)