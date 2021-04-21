





import numpy as np
import torch



class SimSamples:
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
        try:
            self.data = self.data[:,:7]
        except:
            pass

        self.step_indices = np.unique(self.data[:,0])
        self.agents = np.unique(self.data[:,1])

        self.n_steps = len(self.step_indices)
        self.n_agents = len(self.agents)
        self.n_dims = 2

        self.data.requires_grad = False


class SimStateToOneAgentStepSamples (SimSamples):
    '''
    Loads data files and yields samples consisting of:
        x:  full state at time step n
        y:  position of agent m at time step n+1
    where n and m are combined deterministically to form i, the index of the sample.
    '''

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
        Returns x, y where
            x = (agent_id, full state at time t)
            y = agent state at time t+1
        '''

        return (
            (
                n % self.n_agents,
                self.data[
                    int(n/self.n_agents) * self.n_agents : int(n/self.n_agents) * self.n_agents + self.n_agents,
                    :
                    ]
            ),
            self.data[
                self.n_agents + n,
                3:5
            ]
        )

class SimICsToFullAgentTrajectorySamples (SimSamples):
    '''
    Loads data files and yields samples consisting of:
        x:  initial conditions (full state at time step 0)
        y:  full trajectory of agent m
    indexed by m.
    '''

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
            (   
                torch.tensor(n % self.n_agents, requires_grad=False),
                self.data[0:self.n_agents,:],
            ),
            self.data[
                n+self.n_agents:self.n_steps*self.n_agents:self.n_agents,
                3:5
            ]
        )

class SimStateToSimState (SimSamples):
    def __len__(self):
        '''
        Returns the number of samples (n_steps - 1)
        '''

        return self.n_steps - 1

    def __getitem__(self, n):
        '''
        Gets the kinematic variables from two states, at n and n+1, flattened into vectors.
        '''

        return (
            np.reshape(
                    self.data[
                    n * self.n_agents : (n+1) * self.n_agents,
                    3:7
                ],
                -1
            ),
            np.reshape(
                self.data[
                    (n+1) * self.n_agents : (n+2) * self.n_agents,
                    3:7
                ],
                -1
            )
        )