import numpy as np
import torch


class ReplayBuffer( object ):
	"""
	A buffer for storing and sampling experience tuples for off-policy reinforcement learning algorithms.

	Attributes:
		max_size (int): Maximum number of transitions to store in the buffer.
		ptr (int): Pointer to the current position for inserting new transitions.
		size (int): Current number of transitions stored in the buffer.
		state, action, next_state, reward, not_done (np.ndarray): Arrays for storing experience components.
		device (torch.device): Device to which sampled tensors are moved.
	"""

	def __init__( self, state_dim, action_dim, max_size, device ):
		"""
		Constructor of the replay buffer.

		Args:
			state_dim (int): Dimension of the state space.
			action_dim (int): Dimension of the action space.
			max_size (int, optional): Buffer size (maximum number of transitions that can be stored). Defaults to 1e6.
		"""
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros( ( max_size, state_dim ) )
		self.action = np.zeros( ( max_size, action_dim ) )
		self.next_state = np.zeros( ( max_size, state_dim ) )
		self.reward = np.zeros( ( max_size, 1 ) )
		self.not_done = np.zeros( ( max_size, 1 ) )

		self.device = device


	def add( self, state, action, next_state, reward, done ):
		"""
		Add a transition sample to the buffer.

		Args:
			state (np.ndarray): Current state.
			action (np.ndarray): Action taken.
			next_state (np.ndarray): Next state after action.
			reward (float): Reward received.
			done (bool or float): Indicator if episode ended after this transition.
		"""
		self.state[ self.ptr ] = state
		self.action[ self.ptr ] = action
		self.next_state[ self.ptr ] = next_state
		self.reward[ self.ptr ] = reward
		self.not_done[ self.ptr ] = 1. - done

		self.ptr = ( self.ptr + 1 ) % self.max_size
		self.size = min( self.size + 1, self.max_size )


	def sample( self, batch_size ):
		"""
		Sample a batch of transitions from the buffer.

		Args:
			batch_size (int): Number of transitions to sample.

		Returns:
			Tuple[torch.FloatTensor]: Batch of (state, action, next_state, reward, not_done) tensors.
		"""
		ind = np.random.randint( 0, self.size, size=batch_size )

		return (
			torch.FloatTensor( self.state[ ind ]).to( self.device ),
			torch.FloatTensor( self.action[ ind ]).to( self.device ),
			torch.FloatTensor( self.next_state[ ind ] ).to( self.device ),
			torch.FloatTensor( self.reward[ ind ] ).to( self.device ),
			torch.FloatTensor( self.not_done[ ind ] ).to( self.device )
		)