import numpy as np
import sys, os, subprocess

if len(list(sys.argv)) not in (3, 4, 5):
    print("Usage: gravity_testing_suite.py sampling_width sample_count [optimizer] [dataset]")
    sys.exit(1)

sampling_width = float(sys.argv[1])
sample_count = int(sys.argv[2])
true_value = 1

sampled_parameter_space = np.random.uniform(
    low=(true_value - sampling_width),
    high=(true_value + sampling_width),
    size=sample_count
)

if len(sys.argv) > 3:
    optimizer = sys.argv[3]
    if len(sys.argv) > 4:
        dataset = str(sys.argv[3])
    else:
        dataset = None
else:
    optimizer = None

print(f'Spawning {sample_count} training processes...')
sub_procs = []
for G_guess in sampled_parameter_space:
    if optimizer != None:
        if dataset != None:
            sub_procs.append(subprocess.Popen(["python", "learn_v5.py", str(G_guess), str(optimizer), dataset]))
        else:
            sub_procs.append(subprocess.Popen(["python", "learn_v5.py", str(G_guess), str(optimizer)]))
    else:
        sub_procs.append(subprocess.Popen(["python", "learn_v5.py", str(G_guess)]))

print("Spawned.")
[p.wait() for p in sub_procs]