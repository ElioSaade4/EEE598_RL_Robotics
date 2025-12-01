import numpy as np
from dm_control import suite


def eval_policy( agent, domain, task, seed, eval_episodes=5 ): 
	"""
	Runs episodes to evaluate a given policy in a specific environment.

	Args:
		agent: policy/agent to evaluate.
		domain (str): The domain name for dm_control.suite.load.
		task (str): The task name within the domain.
		seed (int): Random seed for environment initialization.
		eval_episodes (int, optional): Number of episodes to run for evaluation. Defaults to 10.

	Returns:
		avg_reward (float): Average return over the evaluation episodes.
		std_eval (float): Standard deviation of returns over the evaluation episodes.

	"""

	# Initialize environment
	environment_kwargs = { 'flat_observation': True }
	eval_env = suite.load( domain_name = domain, 
						  task_name = task, 
						  environment_kwargs = environment_kwargs, 
						  task_kwargs = {'random': seed  } )

	eval_rewards = []

	for _ in range( eval_episodes ):
		episode_reward = 0.
		state_type, reward, discount, state = eval_env.reset()
		done = False

		while not done:
			action = agent.evaluate( state[ 'observations' ] )
			step_type, reward, discount, state = eval_env.step( action )
			done = step_type.last()
			episode_reward += reward

			if done:
				eval_rewards.append( episode_reward )

	avg_reward = sum( eval_rewards ) / eval_episodes
	std_eval = np.std( eval_rewards )

	print( "---------------------------------------" )
	print( f"Evaluation over {eval_episodes} episodes -- average: {avg_reward:.3f} -- std: {std_eval:.3f}" )
	print( "---------------------------------------" )

	return avg_reward, std_eval