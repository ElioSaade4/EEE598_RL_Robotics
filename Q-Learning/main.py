import random
import numpy as np
import matplotlib.pyplot as plt

from QLearning import QLearning
from Gridworld import Gridworld


def evaluate_policy( agent, n_episodes ):
    """
    Evaluates the learned policy by running several episodes without exploration.

    Args:
        agent ( QLearning ): trained QLearning agent
        n_episodes ( int ): number of evaluation episodes

    Returns:
        avg_reward ( float ): average reward over the evaluation episodes
    """
    total_rewards = 0

    for _ in range( n_episodes ):
        state = env.reset()
        done = False
        episode_rewards = 0
        steps = 0

        while not done and steps < 25:
            steps += 1
            action = np.argmax( agent.Q[ agent.stateToIndex( state ), : ] )   # Select greedy action
            next_state, reward, done = env.step( action )
            episode_rewards += reward
            state = next_state

        total_rewards += episode_rewards

    avg_reward = total_rewards / n_episodes
    return avg_reward


if __name__ == '__main__':

    # Training parameters 
    alpha = 0.1                     # learning rate
    gamma = 0.99                    # discount factor
    eps = 0.05                      # exploration rate
    n_trials = 10                   # number of trials
    n_episodes = 1000               # number of episodes per trial
    eval_interval = 20              # evaluate the policy every eval_interval episodes

    plt.figure()


    for trial in range( n_trials ):
        print( f'Trial { trial + 1 } / { n_trials }' )
        print( '------------' )

        seed = trial
        np.random.seed( seed )
        random.seed( seed )

        # Initialize environment and agent
        env = Gridworld()
        agent = QLearning( state_dim=9, 
                           action_dim=4,
                           learning_rate=alpha,
                           discount_factor=gamma,
                           epsilon=eps )

        
        # Variables to store results
        training_rewards = np.zeros( n_episodes )   # rewards per training episode
        eval_rewards = np.zeros( n_episodes // eval_interval )  # average rewards per evaluation

        for episode in range( n_episodes ):
            state = env.reset()
            episode_rewards = 0
            done = False

            steps = 0
            
            while not done and steps < 25:
                steps += 1

                action = agent.selectAction( state )
                next_state, reward, done = env.step( action )
                agent.train( state, action, next_state, reward, done )

                episode_rewards += reward
                state = next_state

                if ( episode + 1 ) % eval_interval == 0:
                    index = ( ( episode + 1 ) // eval_interval ) - 1
                    eval_rewards[ index ] = evaluate_policy( agent, n_episodes=10 )

            training_rewards[ episode ] = episode_rewards
            # print( f'Episode { episode + 1 } / { n_episodes_training }, Reward: { episode_rewards }' )

        plt.plot( range( eval_interval, n_episodes + 1, eval_interval ), eval_rewards, label=f'Seed { seed }' )
        
        agent.print_greedy_policy()
        print()
    
    plt.legend()
    plt.xlabel( 'Episode')
    plt.ylabel( 'Evaluation Reward' )
    plt.grid()
    plt.title( f'Q-Learning: alpha={alpha}, gamma={gamma}, eps={eps}' )
    plt.show()