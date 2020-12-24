





import numpy as np
import torch






class SimSamples:
    '''
    '''

    def __init__(self, path):
        '''
        '''

        self.data = torch.tensor(
            np.loadtxt(
                path,
                delimiter=',',
                skiprows=1
            ),
            requires_grad=False
        )

        self.step_indices = np.unique(self.data[:,6])
        self.agents = np.unique(self.data[:,0])

        self.n_steps = len(self.step_indices)
        self.n_agents = len(self.agents)
        self.n_dims = 2

    def __len__(self):
        '''
        '''

        return \
            len(self.data) - self.n_agents

    def __getitem__(self, n):
        '''
        Loops through the data in two tiers:
            - every timestep
            - every agent within that timestep
        '''

        return (
            self.data[
                int(n/self.n_agents) * self.n_agents : int(n/self.n_agents) * self.n_agents + self.n_agents,
                :
                ],
            self.data[
                self.n_agents + n,
                1:3
            ]
        )