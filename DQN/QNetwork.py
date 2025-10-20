import torch.nn as nn
import torch.nn.functional as F


class QNetwork( nn.Module ):

    def __init__( self, state_dim, num_actions, layer_size ):
        """Initializes a Q-network for discrete actions. 
           It takes state as input and outputs the Q-values for each action.

        Args:
            state_dim (int): Dimension of the state space
            num_actions (int): Number of possible discrete actions
            layer_size (int): Size of the hidden layers
        """
        super( QNetwork, self ).__init__()
        self.fc1 = nn.Linear( state_dim, layer_size )
        self.fc2 = nn.Linear( layer_size, layer_size )
        self.fc3 = nn.Linear( layer_size, num_actions )


    def forward( self, state ):
        """Performs a forward pass in the network and returns the Q-values for all the possible actions

        Args:
            state (tensor): Input state tensor.

        Returns:
            q-values (tensor): Q-values for all possible actions.
        """
        x = F.relu( self.fc1( state ) )
        x = F.relu( self.fc2( x ) )
        q_values = self.fc3( x )
        return q_values
