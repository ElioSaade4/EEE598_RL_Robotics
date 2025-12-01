import os
import torch
import csv
import pyexcelerate
import random
import numpy as np
from dm_control import suite

from TD3 import TD3
from utils import eval_policy


if __name__ == "__main__":

	domain = "cartpole"
	task = "balance"

	eval_seed = 10
	n_trials = 3
	start_timesteps = int( 50e3 )
	max_timesteps = int( 500e3 )
	eval_freq = int( 10e3 )
	batch_size = 256
	buffer_size = int( 1e6 )
	discount = 0.99
	tau = 0.005
	actor_lr = 1e-3
	critic_lr = 1e-3

	expl_noise = 0.1
	policy_noise = 0.2
	noise_clip = 0.5	
	policy_freq = 2


	# Directory to save results
	results_dir = "./results"

	if not os.path.exists( results_dir ):
		os.makedirs( results_dir )

	model_name = f"TD3_{ domain }_{ task }"
	results_name = os.path.join( results_dir, model_name )

	
	# Iterate over all training trials (different mother random seeds)
	for trial in range( n_trials ):

		# set random seeds	
		seed = trial 

		random.seed( seed )
		np.random.seed( seed ) 
		torch.manual_seed( seed ) 


		print( "-------------------------------------------------------------------------------------------------------" )
		print( f"Policy: TD3, Domain: { domain }, Task: { task }, Seed: { seed }" )
		print( "-------------------------------------------------------------------------------------------------------" )


		model_file = f"{ model_name }_{ seed }"


		# Initialize environment
		environment_kwargs = { 'flat_observation': True }

		env = suite.load( domain_name = domain, 
						  task_name = task, 
						  environment_kwargs = environment_kwargs, 
						  task_kwargs = {'random': ( seed ) } )
		
		# Get information about state and action space
		state_dim = env.observation_spec()[ 'observations' ].shape[ 0 ]
		action_dim = env.action_spec().shape[ 0 ] 
		max_action = float( env.action_spec().maximum[ 0 ] )
		min_action = float( env.action_spec().minimum[ 0 ] )
		action_shape = env.action_spec().shape


		# Agent arguments
		kwargs = {
			"state_dim": state_dim,
			"action_dim": action_dim,
			"max_action": max_action,
			"start_timesteps": start_timesteps,
			"max_timesteps": max_timesteps,
			"discount": discount,
			"tau": tau,
			"actor_lr": actor_lr,
			"critic_lr": critic_lr,
			"batch_size": batch_size,
			"buffer_size": buffer_size,
			"policy_noise": policy_noise,
			"noise_clip": noise_clip,
			"policy_freq": policy_freq
		}
	
		# Initialize agent
		agent = TD3( **kwargs )


		# Variables to keep track of training and to print out results
		episode_timesteps = 0
		episode_num = 0
		episode_reward = 0
		training_rewards = []
		csv_data = [ ( 'Episode Number', 'Training Mean Reward', 'Training Std', 'Evaluation Mean Reward', 'Evaluation Std' ) ]
		

		# Reset environment
		state_type, reward, discount, state = env.reset()
		done = False
		

		# Main training loop 
		for t in range( int( max_timesteps ) ):
			
			episode_timesteps += 1
	
			# Select action randomly or according to policy
			if t < start_timesteps:
				action = np.random.uniform( low=min_action, high=max_action, size=action_shape )
			else:
				action = ( agent.select_action( state[ 'observations' ] )
						   + np.random.normal( 0, max_action * expl_noise, size=action_dim )
							).clip( -max_action, max_action )
	
			# Perform action
			step_type, reward, discount, next_state = env.step( action ) 
			done = step_type.last()

			# Run agent training step
			agent.train( t, state[ 'observations' ], action, next_state[ 'observations' ], reward, done )
	
			state = next_state
			episode_reward += reward
	
			if done:
				training_rewards.append( episode_reward )

				print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
				
				state_type, reward, discount, state = env.reset()
				done = False
				episode_reward = 0
				episode_timesteps = 0
				episode_num += 1


			# Evaluate episode
			# Evaluate with worker 0 because all the networks are synced and the mother seed is used for evaluation
			if  ( t + 1 ) % eval_freq == 0 :
				eval_avg_reward, eval_std = eval_policy( agent, domain, task, eval_seed, eval_episodes=5 )
				training_avg_reward = np.array( training_rewards ).mean()
				training_std = np.std( np.array( training_rewards ) )
				training_rewards = []
				csv_data.append( ( ( t + 1 ), training_avg_reward, training_std, eval_avg_reward, eval_std ) )
		

		# Write results to csv file after each seed is done
		with open( f"{ results_name }_{ seed }.csv", "w", newline="" ) as csvfile:
			writer = csv.writer( csvfile, lineterminator = '\n' )
			writer.writerows( csv_data )

	
	# Combine all results into a single excel file after all trials are done
	wb = pyexcelerate.Workbook()
	for trial in range( n_trials ):
		seed = trial
		results_file = f"{ results_name }_{ seed }.csv"

		# Read csv data
		csv_data = []
		with open( results_file, "r" ) as csvfile:
			reader = csv.reader( csvfile )
			for row in reader:
				csv_data.append( row )

		# Create a new sheet in the excel file and store the results of the current seed
		ws = wb.new_sheet( f"Seed { seed }", data=csv_data )
		os.remove( results_file )		
		
	wb.save( f"{ results_name }.xlsx" )