import torch
import torch.optim as optim
import torch.nn as nn
import math, random, os, glob
import scipy
import numpy as np
from tqdm import tqdm
from multi_agent_kinetics import serialize
from hts.learning import models, data_loaders, tb_logging
from torch.utils.tensorboard import SummaryWriter

torch.autograd.set_detect_anomaly(True)

writer = SummaryWriter()

model = models.KernelLearner(
    input_size=2,
    hidden_size=2,
    output_size=1)

##criterion = nn.MSELoss()
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9
)

true_kernel = lambda x: x/2

cum_loss = 0

##hidden = model.init_hidden()
for e in range(5):
    for i in tqdm(range(5000)):

        optimizer.zero_grad()

        data = (torch.rand(size=(2,), dtype=torch.float, requires_grad=True) - 0.5) * 20
        y_pred = model(data)
                
        y = true_kernel(
            torch.norm(data)
        )

        loss = torch.abs(y_pred - y)
        loss.backward()
        optimizer.step()
        cum_loss = cum_loss + loss

    writer.add_scalar("Loss/train/cum_avg", cum_loss / 5000 * (e+1), e)
    writer.add_scalar("Loss/train/test33", torch.abs(model(torch.Tensor([3,3])) - true_kernel(torch.norm(torch.Tensor([3,3])))), e)
    print(cum_loss / (5000 * (e+1)))
    print(torch.abs(model(torch.Tensor([3,3])) - true_kernel(torch.norm(torch.Tensor([3,3])))))
    

    

writer.flush()