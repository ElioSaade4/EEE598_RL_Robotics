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

    # Choose experiment configuration
    config = 0      # UNCOMMENT line 84
    # config = 1      # UNCOMMENT line 85
    # config = 2      # UNCOMMENT line 86
    # config = 3      # UNCOMMENT line 87

    match config:
        case 0:
            # constant hyperparameters and 10 random seeds
            alpha = 0.1               
            gamma = 0.99             
            eps = 0.05               
            n_trials = 10 
        case 1:
            # 1 random seeds and varying learning rate
            alpha_values = [ 0.01, 0.1, 0.4, 0.9 ]
            gamma = 0.99             
            eps = 0.05               
            n_trials = 1
        case 2:
            # 1 random seeds and varying discount factor
            alpha = 0.1
            gamma_values = [ 0.3, 0.7, 0.99 ]
            eps = 0.05
            n_trials = 1
        case 3:
            # 1 random seeds and varying exploration rate
            alpha = 0.1
            gamma = 0.99
            eps_values = [ 0.001, 0.1, 0.7 ]
            n_trials = 1
                            

    # Other training parameters
    max_steps = 25
    eval_episodes = 10    
    n_episodes = 1000               # number of episodes per trial
    eval_interval = 20              # evaluate the policy every eval_interval episodes

    plt.figure()

    for _ in range( 1 ):            # config = 0
    # for alpha in alpha_values:    # config = 1
    # for gamma in gamma_values:    # config = 2
    # for eps in eps_values:        # config = 3
        for trial in range( n_trials ):
            print( f'Trial { trial + 1 } / { n_trials }' )
            print( '------------' )

            seed = trial
            np.random.seed( seed )
            random.seed( seed )

            # Initialize environment and agent
            env = Gridworld( goal=1, penalty=-1 )
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
                
                while not done and steps < max_steps:
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

            match config:
                case 0:
                    plt.plot( range( eval_interval, n_episodes + 1, eval_interval ), eval_rewards, label=f'Seed { seed }' )
                    plt.title( f'Q-Learning: alpha={alpha}, gamma={gamma}, eps={eps}' )
                case 1:
                    plt.plot( range( eval_interval, n_episodes + 1, eval_interval ), eval_rewards, label=f'alpha={alpha}' )
                    plt.title( f'Q-Learning: gamma={gamma}, eps={eps}' )
                case 2:
                    plt.plot( range( eval_interval, n_episodes + 1, eval_interval ), eval_rewards, label=f'gamma={gamma}' )
                    plt.title( f'Q-Learning: alpha={alpha}, eps={eps}' )
                case 3:
                    plt.plot( range( eval_interval, n_episodes + 1, eval_interval ), eval_rewards, label=f'eps={eps}' )
                    plt.title( f'Q-Learning: alpha={alpha}, gamma={gamma}' )

            agent.print_greedy_policy()
            print()
    
    plt.legend()
    plt.xlabel( 'Episode')
    plt.ylabel( 'Evaluation Reward' )
    plt.grid()
    plt.show()