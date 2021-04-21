import torch
import torch.optim as optim
import torch.nn as nn
import math, random, os, glob
from tqdm import tqdm
from multi_agent_kinetics import serialize
from hts.learning import models, data_loaders, tb_logging
from torch.utils.tensorboard import SummaryWriter

torch.autograd.set_detect_anomaly(True)

writer = SummaryWriter()

## Get data
path = random.choice(
    serialize.list_worlds(
        random.choice(
            glob.glob('./data/*/')
        )
    )
)
print(f"Choosing world {path}")
loaded_params = serialize.load_world(path)[1]

model = models.KernelLearner(
    input_size=2,
    hidden_size=2,
    big_layers_size=42,
    output_size=2)

criterion = nn.MSELoss()
optimizer = optim.SGD(
    model.parameters(),
    lr=0.005
)

hidden = model.init_hidden()
i = 0

for epoch in tqdm(range(5)):

    data = data_loaders.SimStateToOneAgentStepSamples(path)

    for sample in data:

        if type(sample) == tuple:

            for row in range(sample[0][1].shape[0]):
                data = torch.tensor(sample[0][1][row,3:5]).float()
                y_pred, hidden = model(data, hidden)
            
            y = sample[1].float()
            y_pred = y_pred.float()

            loss = criterion(y_pred, y)
            writer.add_scalar("Loss/train", loss, i)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            i = i + 1

writer.flush()