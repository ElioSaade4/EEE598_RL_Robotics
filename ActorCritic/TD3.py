import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from ReplayBuffer import ReplayBuffer

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor( nn.Module ):
	def __init__( self, state_dim, action_dim, max_action ):
		"""Constructor of the actor network. Input: states. Ouput: actions.

        Args:
            state_dim ( int ): Number of states, considering a flattened state vector
            action_dim ( int ): Number of actions
            max_action ( np.ndarray ): Vector of maximum value for each action
        """ 
		super( Actor, self ).__init__()

		self.l1 = nn.Linear( state_dim, 256 )
		self.l2 = nn.Linear( 256, 256 )
		self.l3 = nn.Linear( 256, action_dim )
		
		self.max_action = max_action
		

	def forward( self, state ):
		"""Performs a forward pass through the actor network to compute actions.

        Args:
            state (torch.Tensor): Batch of state vectors.

        Returns:
            torch.Tensor: Batch of action vectors.
        """
		a = F.relu( self.l1( state ) )
		a = F.relu( self.l2( a ) )
		return self.max_action * torch.tanh( self.l3( a ) )


class Critic( nn.Module ):
	def __init__( self, state_dim, action_dim ):    
		"""Constructor of the critic network. Input: states and actions. Output: Q-value.

        Args:
            state_dim (int): Dimension of the state space, considering a flattened state vector.
            action_dim (int): Dimension of the action space.
        """
		super( Critic, self ).__init__()

		# Q1 architecture
		self.l1 = nn.Linear( state_dim + action_dim, 256 )
		self.l2 = nn.Linear( 256, 256 )
		self.l3 = nn.Linear( 256, 1 )

		# Q2 architecture
		self.l4 = nn.Linear( state_dim + action_dim, 256 )
		self.l5 = nn.Linear( 256, 256 )
		self.l6 = nn.Linear( 256, 1 )


	def forward( self, state, action ):
		"""Performs a forward pass through both critic networks.

        Args:
            state (torch.Tensor): Batch of state vectors.
            action (torch.Tensor): Batch of action vectors.

        Returns:
            torch.Tensor: Estimated Q-values for the given state-action pairs.
        """
		sa = torch.cat( [ state, action ], dim=1 )

		q1 = F.relu( self.l1( sa ) )
		q1 = F.relu( self.l2( q1 ) )
		q1 = self.l3( q1 )

		q2 = F.relu( self.l4( sa ) )
		q2 = F.relu( self.l5( q2 ) )
		q2 = self.l6( q2 )

		return q1, q2



class TD3( object ): 
	def __init__( self,
				  state_dim,
				  action_dim,
				  max_action,
				  start_timesteps,
				  max_timesteps,
				  discount,
				  tau,
				  actor_lr,
				  critic_lr,
				  policy_noise,
				  noise_clip,
				  policy_freq,
				  batch_size,
				  buffer_size
				  ):
		"""Constructor of the TD3 agent.

		Args:
			state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            max_action (float): Maximum action value (for action scaling).
            start_timesteps (int): Number of timesteps to collect random actions.
            max_timesteps (int): Maximum number of timesteps to run the agent.
            discount (float): Discount factor for future rewards.
            tau (float): Soft update factor for target networks.
            actor_lr (float): Learning rate for the actor network.
            critic_lr (float): Learning rate for the critic networks.
			policy_noise (float): Standard deviation of Gaussian noise added for target policy smoothing.
			noise_clip (float): Maximum absolute value for target policy smoothing noise.
			policy_freq (int): Frequency of delayed policy updates.
            batch_size (int): Batch size for training.
            buffer_size (int): Size of the replay buffer.
        """
		self.device = torch.device( "cpu" )

		# Actor Network
		self.actor = Actor( state_dim, action_dim, max_action ).to( self.device )
		self.actor_target = copy.deepcopy( self.actor )
		self.actor_optimizer = torch.optim.Adam( self.actor.parameters(), lr=actor_lr )

		# Critic Network
		self.critic = Critic( state_dim, action_dim ).to( self.device )
		self.critic_target = copy.deepcopy( self.critic )
		self.critic_optimizer = torch.optim.Adam( self.critic.parameters(), lr=critic_lr )

		# Experience replay buffer and LNSS buffer
		self.replay_buffer = ReplayBuffer( state_dim, action_dim, max_size=buffer_size, device=self.device )

		# Training parameters
		self.max_action = max_action
		self.start_timesteps = start_timesteps
		self.max_timesteps = max_timesteps
		self.discount = discount
		self.tau = tau
		self.batch_size = batch_size
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq


	def update_networks( self, t ):
		"""Train the TD3 actor and critic networks using experiences from the replay buffer.

        Args:
            t (int): Global timestep index (used for delayed policy updates).
		"""

		# Sample replay buffer 
		state, action, next_state, reward, not_done = self.replay_buffer.sample( self.batch_size )

		# Compute the target Q value
		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = ( torch.randn_like( action ) * self.policy_noise ).clamp( -self.noise_clip, self.noise_clip )
			next_action = ( self.actor_target( next_state ) + noise ).clamp( -self.max_action, self.max_action )

			target_Q1, target_Q2 = self.critic_target( next_state, next_action )
			min_target_Q = torch.min( target_Q1, target_Q2 )
	
			target_Q = reward + ( not_done * self.discount * min_target_Q )

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic( state, action )

		# Compute critic loss
		critic_loss = F.mse_loss( current_Q1, target_Q ) + F.mse_loss( current_Q2, target_Q )

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if t % self.policy_freq == 0:
			
			# For actor updates, gradient calculations are not required for critic parameters, but gradients need to flow for the actor parameters
			for param in self.critic.parameters():
				param.requires_grad_( False )
			
			# Compute actor loss using Q1
			current_Q1, _ = self.critic( state, self.actor( state ) )
			actor_loss = -current_Q1.mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			for param in self.critic.parameters():
				param.requires_grad_( True )

			# Soft update target networks
			for param, target_param in zip( self.critic.parameters(), self.critic_target.parameters() ):
				target_param.data.copy_( self.tau * param.data + ( 1 - self.tau ) * target_param.data )

			for param, target_param in zip( self.actor.parameters(), self.actor_target.parameters() ):
				target_param.data.copy_( self.tau * param.data + ( 1 - self.tau ) * target_param.data )


	def train( self, t, state, action, next_state, reward, done ):
		"""The main training function for the agent at every timestep.

        Args:
            t (int): Global timestep (used for delayed policy updates).
            state (np.ndarray): Current observation (flattened) at time t.
            action (np.ndarray): Action taken at time t.
            next_state (np.ndarray): Next observation after the action.
            reward (float): Reward received after taking `action` in `state`.
            done (bool): Whether the episode terminated after this transition.
        """
		# Append the single-step transition tuple to the experience buffer
		self.replay_buffer.add( state, action, next_state, reward, done )

		# Train the agent after the initial random phase
		if t >= self.start_timesteps:
			self.update_networks( t )


	def select_action( self, state ):
		"""Interface function to select actions in the main training episodes.

        Args:
            state (numpy.ndarray): Current state observation

        Returns:
            numpy.ndarray: Selected action vector
        """
		state = torch.FloatTensor( state.reshape( 1, -1 ) ).to( self.device )

		with torch.no_grad():
			action = self.actor( state ).cpu().data.numpy().flatten()

		return action
	
	
	def evaluate( self, state ):
		"""Interface function to select actions in the evaluation episodes.
		   For TD3, it is a wrapper around select_action because the actor is deterministic.

		Args:
            state (numpy.ndarray): Current state observation

        Returns:
            action(numpy.ndarray): Selected action vector
        """
		return self.select_action( state )


	def save( self, filename ):
		"""Saves the current state of the agent's networks and optimizers.

        Args:
            filename (str): Base filename to save the model states
        """
		torch.save( self.critic.state_dict(), filename + "_critic" )
		torch.save( self.critic_optimizer.state_dict(), filename + "_critic_optimizer" )
		
		torch.save( self.actor.state_dict(), filename + "_actor" )
		torch.save( self.actor_optimizer.state_dict(), filename + "_actor_optimizer" )


	def load( self, filename ):
		"""
        Loads the saved state of the agent's networks and optimizers.

        Args:
            filename (str): Base filename from which to load the model states. 
            				The function expects files with suffixes '_critic', '_critic_optimizer', '_actor', and '_actor_optimizer'.
        """
		self.critic.load_state_dict( torch.load( filename + "_critic" ) )
		self.critic_optimizer.load_state_dict( torch.load( filename + "_critic_optimizer" ) )
		self.critic_target = copy.deepcopy( self.critic )

		self.actor.load_state_dict( torch.load( filename + "_actor" ) )
		self.actor_optimizer.load_state_dict( torch.load( filename + "_actor_optimizer" ) )
		self.actor_target = copy.deepcopy( self.actor )
		