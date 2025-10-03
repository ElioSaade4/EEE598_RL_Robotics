import random
import numpy as np


class QLearning:

    def __init__( self, 
                  state_dim, 
                  action_dim, 
                  learning_rate=0.1, 
                  discount_factor=0.99, 
                  epsilon=0.1 ):
        """
        Constructor of the QLearning agent class.
        """
        self.Q = np.zeros( ( state_dim, action_dim ) )      # Matrix of Q values. Rows represent states and column represent actions.

        # Training parameters
        self.alpha = learning_rate    
        self.gamma = discount_factor   
        self.epsilon = epsilon     


    def selectAction( self, state ):
        """
        Selects an action from state s using the Epsilon Greedy algorithm

        Args:
            s ( int ): index of current state. Must be between 0 and 25.

        Returns:
            selected_action ( int ): index of the selected action ( between 0 and 3 )
        """
        s_index = self.stateToIndex( state )       
        
        # Find the indices of greedy and non-greedy actions
        max_Q = np.max( self.Q[ s_index, : ] )    # max action value for the state s
        greedy_actions = np.where( self.Q[ s_index, : ] == max_Q )[ 0 ]
        non_greedy_actions = np.where( self.Q[ s_index, : ] != max_Q )[ 0 ]

        # Select greedy action with probabiltiy 1-eps and non-greedy action with probability eps
        selection = random.random()

        if selection > self.epsilon or len( non_greedy_actions ) == 0:              
            # select non-greedy action with equal probabilities (if multiple ones have maximum Q)
            selected_action = random.choice( greedy_actions )

        else:       
            # Select non-greedy action with equal probabilities
            selected_action = random.choice( non_greedy_actions )

        return selected_action
    

    def stateToIndex( self, state ):
        """
        Converts a (row, column) state representation to a single integer index.

        Args:
            state ( tuple ): current state as (row, column)

        Returns:
            index ( int ): index of the state between 0 and 24
        """
        row, col = state

        match row:
            case 0:
                index = col
            case 1:
                if col == 0:
                    index = 4
                else:
                    index = 5
            case 2:
                index = col + 6

        return index

    
    def train( self, state, action, next_state, reward, done ):
        """
        Updates the Q table using the Q-Learning update rule.

        Args:
            state ( tuple ): current state as (row, column)
            action ( int ): action taken (0: up, 1: right, 2: down, 3: left)
            next_state ( tuple ): next state as (row, column)
            reward ( float ): reward received after taking the action
            done ( bool ): whether the episode has ended

        """
        s_index = self.stateToIndex( state )

        # Q-Learning update rule
        if not done:
            sp_index = self.stateToIndex( next_state )
            max_Q = np.max( self.Q[ sp_index, : ] ) 
        else:
            max_Q = 0

        self.Q[ s_index, action ] = self.Q[ s_index, action ] + self.alpha * ( reward + self.gamma * max_Q - self.Q[ s_index, action ] )

    
    def print_greedy_policy( self ):
        """
        Displays the greed policy learned by the agent for all states. 
        It shows the action with the highest Q-value for each state.
        (Assumes that the Q table has been learned and is not all zeros.)
        """
        actions = [ 'up', 'right', 'down', 'left' ]
        print( 'Greedy policy:' )
        for i in range( 9 ):
            greedy_action = np.argmax( self.Q[ i, : ] )
            print( f'    State { i }: { actions[ greedy_action ] }' )