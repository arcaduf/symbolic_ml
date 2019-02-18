'''
Plot feature importance
'''

from __future__ import print_function , division
import argparse
import sys , os
import json
import numpy as np




# =============================================================================
# MATPLOTLIB WITH DISABLED X-SERVER 
# =============================================================================

import matplotlib as mpl
mpl.use( 'Agg' )
import matplotlib.pyplot as plt




# =============================================================================
# MY VARIABLE FORMAT 
# =============================================================================

myint    = np.int
myfloat  = np.float32
myfloat2 = np.float64




# =============================================================================
# Get input arguments 
# =============================================================================

def _example():
    print( '\n\nBar plot of feature importance by aggregating one-hot encoded variables:\n' )
    print( '"""\npython syml_fimp_plot.py -i syml_history_rf-class_2018-11-14-20-05-44_16159701171257366993.json -e\n"""\n\n' )


def _get_args():
    parser = argparse.ArgumentParser( description     = 'Bar plot of feature importance from SYML .json history file',
                                      formatter_class = argparse.ArgumentDefaultsHelpFormatter    ,
                                      add_help        = False )
    
    parser.add_argument('-i', '--file_json', dest='file_json',
                        help='Specify input CSV file containing features and output variable')

    parser.add_argument('-e', '--encoding', dest='encoding', action='store_true',
                        help='Aggregate features that were split because of one-hot encoding')

    parser.add_argument('-h', '--help', dest='help', action='store_true',
                        help='Print example command line' )

    args = parser.parse_args()

    if args.help:
        parser.print_help()
        _example()
        sys.exit()

    if args.file_json is None:
        parser.print_help()
        sys.exit('\nERROR: Input SYML JSON history file not specified!\n')

    if os.path.isfile( args.file_json ) is False:
        parser.print_help()
        sys.exit('\nERROR: Input SYML JSON history file does not exist!\n')

    return args




# =============================================================================
# Get data from JSON
# =============================================================================

def _get_data_from_json( file_json ):
    df_json = json.loads( open( file_json ).read() )

    feat_imp   = df_json[ 'feature_importance' ]
    feat_names = df_json[ 'feature_used' ]
    feat_enc   = df_json[ 'feature_cols_enc' ]
    
    n_arr = len( feat_imp )

    feat_imp   = np.array( feat_imp ).reshape( n_arr , len( feat_imp[0] ) )
    feat_names = np.array( feat_names )
    
    return feat_names , feat_imp , n_arr , feat_enc




# =============================================================================
# Aggregate features that got split because of one-hot encoding
# =============================================================================

def _aggregate_features( feat_imp , cols_all , cols_enc ):
    # Transform list into array
    feat_imp = np.array( feat_imp )


    # Get indices of encoded columns among the list of columns
    inds_enc = []

    for i in range( len( cols_enc ) ):
        aux = []
        print( '\nOriginal columns: ', cols_enc[i] )

        for j in range( len( cols_all ) ):
            if cols_enc[i] in cols_all[j]:
                aux.append( j )
                print( '\t---> Encoded column: ', cols_all[j] )
        inds_enc.append( aux )


    # Combine feature importance
    feat_imp_new = [];  feat_names_new = [];  n_col_new = 0

    for i in range( len( cols_all ) ):
        flag = True
        for j in range( len( inds_enc ) ):
            if i in inds_enc[j]:
                flag = False
                break
        
        if flag:
            feat_imp_new.append( feat_imp[:,i] )
            feat_names_new.append( cols_all[i] )
            n_col_new += 1
    
    for i in range( len( inds_enc ) ):
        mean = np.zeros( feat_imp.shape[0] )

        for j in range( len( inds_enc[i] ) ):
            mean += feat_imp[:,inds_enc[i][j]]

        mean /= myfloat( len( inds_enc[i] ) )
        feat_imp_new.append( mean )
        feat_names_new.append( cols_enc[i] )
        n_col_new += 1

    feat_imp_new   = np.array( feat_imp_new ).reshape( feat_imp.shape[0] , n_col_new )
    feat_names_new = np.array( feat_names_new )

    return feat_names_new , feat_imp_new




# =============================================================================
# Plot separate bar plots of feature importance
# =============================================================================

def _plot_separate_bar_plots( feat_names , feat_imp , n_arr  , file_in , width=0.6 ):
    # Main for loop
    for i in range( n_arr ):
        # Initialize figure
        fig  = plt.figure()
        ax   = fig.add_subplot( 111 )
        axes = plt.gca()
    

        # Eliminate frame
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)


        # Get positions for y-labels
        y_pos = np.arange( len( feat_names ) )


        # Set x-ticks
        plt.xticks( fontsize=12, fontweight='bold' )
    
       
        # Get arrays
        feats = np.array( feat_names ).copy()
        imps  = np.array( feat_imp[i] )

        
        # Order features according to their importance
        inds_sort = np.argsort( imps )
        imps      = imps[ inds_sort ]
        feats     = feats[ inds_sort ]

        
        # Establish y-ticks
        plt.yticks( y_pos , feats , fontsize=12 ,
                    fontweight='bold' , rotation=0 )
    
        
        # Add text at the end of each column
        for j, v in enumerate( imps ):
            ax.text( v + 0.1 , j - 0.05 , str( round( v , 4 ) ) , color='red' ,
                     fontsize=12 , fontweight='bold' )

        
        # Bar plot
        plt.barh( y_pos , imps , width , align='center', alpha=0.7 )

        
        # Save figure
        #plt.tight_layout()

        if i < 10:
            str_num = '0' + str( i )
        else:
            str_num = str( i )
        string    = 'fold' + str_num
        
        plt.title( 'Feature importance ' + string , fontsize=16 ,fontweight='bold' )
        
        save_plot = file_in[:len( file_in )-5] + '_feature-importance_' + string + '.png'
        
        if '.svg' in save_plot:
            plt.savefig( save_plot , format='svg' , dpi=1200 )
        else:
            plt.savefig( save_plot )
        
        
        

# =============================================================================
# Plot composite bar plot of feature importance
# =============================================================================

def _plot_composite_bar_plot( feat_names , feat_imp , n_arr  , file_in  , width=0.6 ):
    # Compute mean and std of all features
    feats    = np.array( feat_names  ).copy()
    feat_imp = np.array( feat_imp )

    imps = np.mean( feat_imp , axis=0 )
    std  = np.std( feat_imp , axis=0 )
    

    # Initialize figure
    fig  = plt.figure()
    string = str( n_arr ) + '-fold CV'
    plt.title( string + ' composite feature importance' , fontsize=16 ,fontweight='bold' )
    ax   = fig.add_subplot( 111 )
    axes = plt.gca()
    

    # Eliminate frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)


    # Get positions for y-labels
    y_pos = np.arange( len( feats ) )


    # Set x-ticks
    plt.xticks( fontsize=12, fontweight='bold' )
    

    # Order features according to their importance
    inds_sort = np.argsort( imps )
    imps      = imps[ inds_sort ]
    std       = std[ inds_sort ]
    feats     = feats[ inds_sort ]


    # Establish y-ticks
    plt.yticks( y_pos , feats , fontsize=12 ,
                fontweight='bold' , rotation=0 )
    
        
    # Add text at the end of each column
    aux = imps + std

    for i, v in enumerate( aux ):
        ax.text( v + 0.1 , i - 0.05 , str( round( v , 4 ) ) , color='red' ,
                 fontsize=12 , fontweight='bold' )

        
    # Bar plot
    plt.barh( y_pos , imps , width , align='center', alpha=0.7 )


    # Add error bar
    ax.errorbar( imps , y_pos , xerr=std , ecolor='black' , fmt='go' )
        

    # Save figure
    #plt.tight_layout()

    save_plot = file_in[:len( file_in )-5] + '_feature-importance_composite.png'
        
    if '.svg' in save_plot:
        plt.savefig( save_plot , format='svg' , dpi=1200 )
    else:
        plt.savefig( save_plot )
 
        
     

# =============================================================================
# MAIN
# =============================================================================

def main():
    # Get input arguments
    args = _get_args()
    print( '\nInput JSON history file: ', args.file_json )


    # Get list of ground truth and probabilities
    feat_names , feat_imp , n_arr , feat_enc = _get_data_from_json( args.file_json )
    print( '\nNumber of arrays: ', n_arr )
    print( 'Number of features: ', len( feat_imp[0] ) )
    print( '\nFeature used for modeling:\n', feat_names )
    print( '\nOne-hot encoded features:\n', feat_enc )


    # Aggregate features that were split because of one-hot encoding
    if args.encoding:
        print( '\nAggregating features split due to one-hot encoding ....' )
        feat_names , feat_imp = _aggregate_features( feat_imp , feat_names , feat_enc ) 
        print( 'Number of features: ', len( feat_imp[0] ) )


    # Plot ROC curves
    print( '\nPlot separate bar plots of feature importance ....' )
    _plot_separate_bar_plots( feat_names , feat_imp , n_arr , args.file_json )

    print( '\nPlot composite bar plot of feature importance ....' )
    _plot_composite_bar_plot( feat_names , feat_imp , n_arr , args.file_json )

    print( '\n\n' )




# =============================================================================
# CALL TO MAIN
# =============================================================================

if __name__ == '__main__':
    main()

