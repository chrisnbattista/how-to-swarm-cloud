import numpy as np
import sys, os, subprocess

if len(list(sys.argv)) != 3:
    print("Usage: gravity_testing_suite.py sampling_width sample_count")
    sys.exit(1)

sampling_width = float(sys.argv[1])
sample_count = int(sys.argv[2])
true_value = 1

sampled_parameter_space = np.random.uniform(
    low=(true_value - sampling_width),
    high=(true_value + sampling_width),
    size=sample_count
)

print(f'Spawning {sample_count} training processes...')
for G_guess in sampled_parameter_space:
    subprocess.Popen(["python", "learn_v5.py", str(G_guess)])
print("Spawned.")