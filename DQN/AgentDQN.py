import copy
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from QNetwork import QNetwork
from ReplayBuffer import ReplayBuffer


class AgentDQN():

    def __init__( self, 
                  state_dim, 
                  action_dim, 
                  n_actions,
                  layer_size, 
                  lr, 
                  gamma,
                  tau,
                  buffer_size,
                  batch_size,
                  use_cuda ):
        """Constructor of the DQN agent.

        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            n_actions (int): Number of possible discrete actions
            layer_size (int): Number of neurons in the hidden layers
            lr (float): Learning rate for the neural network optimizer
            gamma (float): Discount factor
            tau (float): Coefficient for soft update of target network
            buffer_size (int): Size of the replay buffer
            batch_size (int): Size of the mini-batch for training
            use_cuda (bool): Whether to use CUDA GPU for training
        """
        # Set the computation device
        self.device = torch.device( "cuda" if torch.cuda.is_available() and use_cuda else "cpu" )

        # Initialize the Q-network and target Q-network
        self.q_network = QNetwork( state_dim, n_actions, layer_size ).to( self.device )
        self.q_network_target = copy.deepcopy( self.q_network )
        self.optimizer = optim.Adam( self.q_network.parameters(), lr=lr )

        # Initialize the replay buffer
        self.replay_buffer = ReplayBuffer( state_dim, action_dim, buffer_size, self.device )

        # Store training hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_actions = n_actions


    def select_greedy_action( self, state ):
        """Performs a forward pass in the Q-network and returns the index of the action that maximizes the Q-value.

        Args:
            state (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Convert state to a pytorch tensor
        state = torch.FloatTensor( state.reshape( 1, -1 ) ).to( self.device )

        # Forward pass to get Q-values
        with torch.no_grad():
            q_values = self.q_network.forward( state ).cpu().data.numpy().flatten()

        # Get the action with the highest Q-value
        greedy_action = np.argmax( q_values )
        return greedy_action
    

    def select_action( self, state, epsilon ):
        """Selects an action based on an epsilon-greedy strategy.

        Args:
            state (np.ndarray): Current state.
            epsilon (float): Exploration rate.

        Returns:
            int: Index of selected action.
        """
        if np.random.rand() < epsilon:
            return np.random.randint( 0, self.n_actions )
        else:
            return self.select_greedy_action( state )


    def store_experience( self, state, action, next_state, reward, done ):
        """Wrapper function to store transitions in the replay buffer

        Args:
            state (np.ndarray): Current state.
			action (np.ndarray): Action taken.
			next_state (np.ndarray): Next state after action.
			reward (float): Reward received.
			done (bool or float): Indicator if episode ended after this transition.
        """
        self.replay_buffer.add( state, action, next_state, reward, done )


    def train( self ):
        """Trains the Q-network using a batch of transitions sampled from the replay buffer.
        """
        # Check if there are enough samples in the replay buffer
        if self.replay_buffer.size < self.batch_size:
            return
        
        # Sample a batch of transitions
        state, action, next_state, reward, not_done = self.replay_buffer.sample( self.batch_size )

        # Compute the target
        target_q_values = self.q_network_target( next_state )
        max_target_q_values, _ = target_q_values.max( dim=1, keepdim=True )
        target_Q = reward + ( not_done * self.gamma * max_target_q_values )

        # Get current Q-values
        current_q_values = self.q_network( state )
        current_Q = current_q_values.gather( 1, action.long() )

        # Compute loss and perform optimization step
        loss = F.mse_loss( current_Q, target_Q )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update of target network
        for param, target_param in zip( self.q_network.parameters(), self.q_network_target.parameters() ):
            target_param.data.copy_( self.tau * param.data + ( 1 - self.tau ) * target_param.data )


    def print_greedy_policy( self ):
        """
        Displays the greed policy learned by the agent for all states. 
        It shows the action with the highest Q-value for each state.
        """
        actions = [ 'up', 'right', 'down', 'left' ]
        print( 'Greedy policy:' )
        for i in range( 9 ):
            greedy_action = self.select_greedy_action( np.array( i ) )
            print( f'    State { i }: { actions[ greedy_action ] }' )