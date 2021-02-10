## TO DO: Remove explicit indexes, change to lookups from object schema





import numpy as np
from . import experiments, integrators





schemas = {
    '2d': ('id', 'b_1', 'b_2', 'm', 'v_1', 'v_2', 't'),
    '3d': ('id', 'm', 'b_1', 'b_2', 'b_3', 'v_1', 'v_2', 'v_3', 't')
}





class World:

    def __init__(self,
                initial_state=None,
                spatial_dims=2,
                n_agents=0,
                n_timesteps=None,
                timestep=0.01,
                forces=[],
                indicators=[],
                integrator=integrators.integrate_rect_world,
                **kwargs):
        '''
        Create a new World object with fixed parameters: forces, indicators, fixed timestep, etc.
        If n_timesteps is left as None, the World can run indefinitely (very inefficient).
        '''

        self.spatial_dims = spatial_dims
        self.schema = schemas[str(spatial_dims)+'d']

        self.n_timesteps = n_timesteps
        if n_timesteps:
            self.fixed_length = True
            self.history = np.empty( (n_timesteps * n_agents, len(self.schema)))
            self.indicator_history = np.empty( (n_timesteps, len(indicators)) )
        else:
            self.fixed_length = False
            self.history = np.empty( (0, len(self.schema)))
            self.indicator_history = np.empty( (0, len(indicators)) )

        self.n_agents = n_agents

        self.forces = forces
        self.indicators = indicators
        self.integrator = integrator

        self.timestep_length = timestep

        self.current_timestep = None
        if initial_state is not None:
            self._add_state_to_history(initial_state)   
            self._compute_indicators()
            self.n_agents = initial_state.shape[0]
    
    def _add_state_to_history(self, new_state):
        '''
        Adds the new_state to the state history of the world and advances the timestep by one.
        '''

        if self.current_timestep is None:
            self.current_timestep = 0
        else:
            self.current_timestep += 1

        if self.fixed_length:
            self.history[
                (self.current_timestep*self.n_agents) : ((self.current_timestep+1)*self.n_agents), :] = new_state
        else:
            self.history = np.concatenate((self.history, new_state))
        
        return self
    
    def _add_state_to_indicators(self, new_indicators):
        '''
        Concatenates new_indicators to the indicators timeseries at index self.current_timestep.
        '''

        if self.fixed_length:
            self.indicator_history[self.current_timestep, :] = new_indicators
        else:
            self.indicator_history = np.concatenate((self.indicator_history, new_indicators))
        
        return self
    
    def get_state(self):
        '''
        Returns a view of the latest entry in the world state history.
        '''

        if self.current_timestep is None:
            return None

        return self.history[(self.current_timestep*self.n_agents) : ((self.current_timestep+1)*self.n_agents), :]
    
    def reset_state(self, new_state):
        '''
        Switches World to 
        '''
        self.fixed_length = False
        self.history = np.empty( (0, len(self.schema)))
        self.indicator_history = np.empty( (0, len(indicators)) )

        self._add_state_to_history(new_state)
        self._compute_indicators()
    
    def get_history(self):
        '''
        '''
        return self.history
    
    def get_history_with_indicators(self):
        '''
        Returns an array representing both state and indicators across entire time span.
        '''
        return \
            np.concatenate(
                (
                    self.history,
                    np.repeat(self.indicator_history, self.n_agents, axis=0)
                ),
                axis=1
            )
    
    def _compute_indicators(self):
        '''
        '''
        indicator_results = np.empty( (1, len(self.indicators)) )
        for j in range(len(self.indicators)):
            indicator_results[0, j] = self.indicators[j](state)
        self._add_state_to_indicators(indicator_results)
    
    def advance_state(self, steps=1):
        '''
        Advances the underlying world state by the specified number of time steps and records the state trajectory.
        '''

        for i in range(steps):

            state = self.get_state().copy() # state is initially old state

            ## Calculate forces
            # Initialize matrix to hold forces keyed to id
            force_matrix = np.zeros ( (state.shape[0], 2) )
            for force in self.forces:
                force_matrix = force_matrix + force(state)
                '''
                convert world state to pairwise distances
                corrupt pairwise distances (sensor emulation, likely gaussian noise)
                corrupt world state (positions) (gaussian noise, GPS)
                call hcl routine with corrupted pairwise distances, positions (on mothership)
                get back estimated world state
                apply sph (centralized) on estimated world state
                '''

            ## Advance the timestep itself
            state[:,6] += self.timestep_length

            ## Integrate forces over timestep
            self.integrator(state, force_matrix, self.timestep_length)

            # state is now new state, so append it to the history and advance the internal
            # timestep counter
            self._add_state_to_history(state)

            self._compute_indicators()