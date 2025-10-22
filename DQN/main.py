import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from AgentDQN import AgentDQN
from Gridworld import Gridworld


def evaluate_policy( agent, n_episodes, max_steps ):
    """
    Evaluates the learned policy by running several episodes without exploration.

    Args:
        agent ( AgentDQN ): trained DQN agent
        n_episodes ( int ): number of evaluation episodes
        max_steps ( int ): maximum steps per episode

    Returns:
        avg_reward ( float ): average reward over the evaluation episodes
    """
    total_rewards = 0

    for _ in range( n_episodes ):
        state = env.reset()
        done = False
        episode_rewards = 0
        steps = 0

        while not done and steps < max_steps:
            steps += 1
            action = agent.select_greedy_action( state )   # Select greedy action
            next_state, reward, done = env.step( action )
            episode_rewards += reward
            state = next_state

        total_rewards += episode_rewards

    avg_reward = total_rewards / n_episodes
    return avg_reward


if __name__ == '__main__':

    plt.figure()

    n_trials = 5
    n_episodes = 3000
    max_steps = 30
    eval_interval = 100

           
    for trial in range( n_trials ):
        print( f'Trial { trial + 1 } / { n_trials }' )
        print( '------------' )

        # Set random seeds for reproducibility
        seed = trial
        np.random.seed( seed )
        random.seed( seed )
        torch.manual_seed( seed )

        # Initialize environment and agent
        env = Gridworld( goal=1, penalty=-1 )

        agent = AgentDQN( state_dim = 2, 
                          action_dim = 1, 
                          n_actions = 4,
                          layer_size = 64, 
                          lr = 0.001, 
                          gamma = 0.99, 
                          tau = 0.01, 
                          buffer_size = 100000, 
                          batch_size = 32, 
                          use_cuda = False )

        
        # Variables to store results
        training_rewards = np.zeros( n_episodes )   # rewards per training episode
        eval_rewards = np.zeros( n_episodes // eval_interval )  # average rewards per evaluation

        trial_steps = 0

        for episode in range( n_episodes ):
            state = env.reset()
            episode_rewards = 0
            done = False

            episode_steps = 0

            while not done and episode_steps < max_steps:
                trial_steps += 1
                episode_steps += 1

                # act randomly for the first 100 steps to populate replay buffer
                if trial_steps <= 100:
                    action = np.random.randint( 0, agent.n_actions )
                else:
                    action = agent.select_action( state , 0.1 )


                next_state, reward, done = env.step( action )

                agent.store_experience( state, action, next_state, reward, done )

                agent.train()

                episode_rewards += reward
                state = next_state

            training_rewards[ episode ] = episode_rewards

            # Evaluate the learned policy at regular intervals
            if ( episode + 1 ) % eval_interval == 0:
                index = ( ( episode + 1 ) // eval_interval ) - 1
                eval_result = evaluate_policy( agent, n_episodes=10, max_steps=max_steps )
                eval_rewards[ index ] = eval_result

            if (episode + 1 ) % 500 == 0:
                print( f'Training episode { episode + 1 }' )

            
        plt.plot( range( eval_interval, n_episodes + 1, eval_interval ), eval_rewards, label=f'Seed { seed }' )
        agent.print_greedy_policy()
        print()
    
    plt.title( 'DQN Evaluation Curve - Slip' )
    plt.legend()
    plt.xlabel( 'Episode')
    plt.ylabel( 'Evaluation Reward' )
    plt.grid()
    plt.savefig( 'Evaluation Curve.png' )
    plt.show()