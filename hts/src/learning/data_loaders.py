





import numpy as np
import torch






class SimStateToOneAgentStepSamples:
    '''
    Loads data files and yields samples consisting of:
        x:  full state at time step n
        y:  position of agent m at time step n+1
    where n and m are combined deterministically to form i, the index of the sample.
    '''

    def __init__(self, path):
        '''
        Loads samples from a data file as specified.
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
        Returns the number of samples ( n_agents * (n_steps - 1) )
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

class SimICsToFullAgentTrajectorySamples:
    '''
    Loads data files and yields samples consisting of:
        x:  initial conditions (full state at time step 0)
        y:  full trajectory of agent m
    indexed by m.
    '''

    def __init__(self, path):
        '''
        Loads samples from a data file as specified.
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
        Returns the number of samples.
        n_samples = n_agents
        '''

        return self.n_agents

    def __getitem__(self, n):
        '''
        Returns one sample at a time, indexed by agent.
        Sample = (All ICs, agent trajectory)
        '''

        return (
            self.data[0,:],
            self.data[
                n + range(0, self.n_steps-1, self.n_agents),
                1:3
            ]
        )