import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


warnings.filterwarnings( "ignore", message="Workbook contains no stylesheet")


def SMA_3( data ):
    """Compute the simple moving average with window size 3 of a 1D array.
       The computed average is centered, i.e., for index i, the average of (i-1, i, i+1) is taken.
       This is similar to pandas rolling with window=3 and center=True, but also handles the boundary cases (1st and last elements).
    """
    sma = np.zeros_like( data )
    sma[ 0 ] = ( data[ 0 ] + data[ 1 ] ) / 2
    sma[ -1 ] = ( data[ -2 ] + data[ -1 ] ) / 2
    for i in range( 1, len( data ) - 1 ):
        sma[ i ] = ( data[ i - 1 ] + data[ i ] + data[ i + 1 ] ) / 3
    return sma


if __name__ == "__main__":

    # Plot parameters
    domain = 'cartpole'
    task = 'balance'
    SMA_enable = True
    x_lim = int( 500e3 )
    x_step = int( 50e3 )
    y_lim = 1000
    y_step = 100
    n_seeds = 3

    filename = f'TD3_{domain}_{task}.xlsx'
    configs = [ "Training", "Evaluation" ]

    for config in configs:

        title = f'{ domain.capitalize() } { task.capitalize() } - { config }'
        figure_name = f'TD3_{domain}_{task}_{config}.png'

        plt.figure( figsize=( 13, 7 ) )

        df = pd.DataFrame()

        if os.path.exists( filename ):
            df_excel = pd.read_excel( filename, sheet_name=None )

            for i in range( n_seeds ):

                sheet = df_excel[ f'Seed {i}' ]

                if i == 0:
                    df[ 'Episode' ] = sheet[ 'Episode Number' ]
                
                col = f'Seed{ i }'

                if SMA_enable:
                    df[ col ] = SMA_3( sheet[ f'{config} Mean Reward' ].to_numpy() )
                else:
                    df[ col ] = sheet[ f'{config} Mean Reward' ] 

            # convert data frame to long format
            df_long = df.melt(  id_vars='Episode', 
                                value_vars=[ f'Seed{i}' for i in range( n_seeds ) ],
                                var_name='Seed',
                                value_name=f'{config} Reward' )
            
            # plot average reward over seeds with 95% confidence interval
            sns.lineplot( data = df_long,
                                x = 'Episode',
                                y = f'{config} Reward',
                                errorbar = ( 'ci', 95 ),
                                label = 'TD3',
                                linewidth = 1.75
                                )
            
        # first parameter is (x,y), ncols to write labels next to each other
        plt.legend( bbox_to_anchor=( 0.5, 1.15 ), loc='upper center' )

        plt.title( title )        # title

        # x-limits and y-limits for the plot
        plt.xlim( 0, x_lim )
        plt.ylim( 0, y_lim )

        # x-ticks and y-ticks to control the size of the grid lines
        plt.xticks( range( 0, x_lim + 1, x_step ) )  
        plt.yticks( range( 0, y_lim + 1, y_step ) )

        # change x-ticks to regular numbers instead of scientific notation
        plt.ticklabel_format(style='plain', axis='x')
        plt.grid()

        plt.savefig( figure_name )