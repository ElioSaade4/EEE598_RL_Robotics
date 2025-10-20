import random

class Gridworld( object ):

    def __init__( self, goal=1, penalty=-1 ):
        """ 
        Constructor for the 3x4 Gridworld environment with slip dynamics.
        The state is represented as (row, column) tuples, where (0,0) is the bottom-left corner. 
        The row gets incremented by going up and the column gets incremented by going right.
        The actions are represented as integers:
            0: Up
            1: Right
            2: Down
            3: Left
        """
        self.n_rows = 3
        self.n_columns = 4
        self.start_state = ( 0, 0 )
        self.wall_state = ( 1, 1 )
        self.goal_state = ( 2, 3 )
        self.penalty_state = ( 1, 3 )
        self.goal_reward = goal
        self.penalty_reward = penalty

        self.state = self.start_state


    def step( self, action ):
        """
        Takes a step in the environment based on the current state and action.

        Args:
            action ( int ): action to take (0: up, 1: right, 2: down, 3: left)
        
        Returns:
            state ( tuple ): current state as (row, column)
            next_state ( tuple ): next state as (row, column)
            reward ( float ): reward received after taking the action
            done ( bool ): whether the episode has ended
        """
        # Slip dynamics: 80% desired action, 10% slip right, 10% slip left
        slip_prob = random.random()

        if slip_prob < 0.1: 
            # Slip to the right with a 10% chance
            action = ( action + 1 ) % 4
        elif slip_prob < 0.2:
            # Slip to the left with a 10% chance
            action = ( action + 3 ) % 4

        state = self.state
        row, col = state

        # Determine next state
        match action:   
            case 0:     # Up
                next_state = ( min( row + 1, self.n_rows - 1 ), col )
            case 1:    # Right
                next_state = ( row, min( col  + 1, self.n_columns - 1 ) )
            case 2:   # Down
                next_state = ( max( row - 1, 0 ), col )
            case 3:    # Left
                next_state = ( row, max( col - 1, 0 ) )

        # Check for wall
        if next_state == self.wall_state:
            next_state = state

        # Reward and Done
        if next_state == self.goal_state:
            reward = self.goal_reward
            done = True
        elif next_state ==  self.penalty_state:
            reward = self.penalty_reward
            done = True
        else:
            reward = -0.04
            done = False

        self.state = next_state

        return self.stateToIndex( next_state ), reward, done
    

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

    
    def reset( self ):
        """
        Resets the environment to the start state.

        Returns:
            state ( tuple ): starting state as (row, column)
        """
        self.state = self.start_state
        return self.stateToIndex( self.state )