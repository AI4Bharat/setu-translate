# child_process.py
from tqdm import tqdm
from time import sleep

# Simulate a process with a progress bar
for i in tqdm(range(10)):
    sleep(0.1)  # Simulating work by sleeping


for i in tqdm(range(20)):
    sleep(0.1)  # Simulating work by sleeping


for i in tqdm(range(30)):
    sleep(0.1)  # Simulating work by sleeping
