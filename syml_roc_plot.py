'''
Composite ROC curve for 5-fold CV
'''

from __future__ import print_function , division
import argparse
import sys , os
import json
import numpy as np
from sklearn import metrics as me
from scipy import interpolate

from syml_metrics import Metrics




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
    print( '\n\nPlot composite ROC curve from a SYML .json history file:\n' )
    print( '"""\npython syml_roc_plot.py -i syml_history_rf-class_2018-11-14-20-05-44_16159701171257366993.json\n"""\n\n' )


def _get_args():
    parser = argparse.ArgumentParser( description     = 'Plot ROC curve from SYML .json history file',
                                      formatter_class = argparse.ArgumentDefaultsHelpFormatter    ,
                                      add_help        = False )
    
    parser.add_argument('-i', '--file_json', dest='file_json',
                        help='Specify input CSV file containing features and output variable')

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

    trues = df_json[ 'y_true' ]
    probs = df_json[ 'y_prob' ]

    n_arr = len( trues )

    return trues , probs , n_arr




# =============================================================================
# Plot separate ROCs
# =============================================================================

def _plot_separate_roc_curves( trues , probs , n_arr  , cmet , file_in ):
    # Initialize figure
    fig = plt.figure()
    string = str( n_arr ) + '-fold CV'
    plt.title( string + ' separate ROCs' , fontsize=16 ,fontweight='bold' )
    plt.xlabel( 'False Positive Rate' , fontsize=16 )
    plt.ylabel( 'True Positive Rate' , fontsize=16 )
    plt.xticks( fontsize=14 );  plt.yticks( fontsize=14 )


    # Plot "toss-coin" line y = x
    x_line = np.linspace( 0 , 1 , 50 )
    y_line = x_line.copy()
    plt.plot( x_line , y_line , '--' , lw=2 , color='black' )
    

    # Main for loop
    for i in range( n_arr ):
        # Get arrays
        y_score = np.array( probs[i] )
        y_true  = np.array( trues[i] )
        
        
        # Compute all metrics
        cmet._compute_metrics( y_true , y_score )


        # Compute ROC AUC score
        roc_auc = me.auc( cmet._fpr , cmet._tpr )
    
    
        # Get confidence interval
        conf_int = cmet._ci_bootstrap_roc( y_true , y_score )
    
    
        # Get Youden's point for fpr and tpr
        if len( cmet._fpr_best ) > 1:
            cmet._fpr_best = cmet._fpr_best[0]

        if len( cmet._tpr_best ) > 1:
            cmet._tpr_best = cmet._tpr_best[0]


        # Create label
        label = 'Fold n.' + str( i ) + \
                ': AUC = ' + str( round( roc_auc , 2 ) ) + \
                ' , 95%CI= [' + str( round( conf_int[1] , 2 ) ) + ',' + \
                str( round( conf_int[0] , 2 ) ) + ']'
        
        
        # Plot ROC curve
        plt.plot( cmet._fpr , cmet._tpr , lw=2 , label=label )

              
    # Save figure
    plt.tight_layout()
    plt.legend( fontsize=10 )
        
    save_plot = file_in[:len( file_in )-5] + '_separate_roc_curve.png'
        
    if '.svg' in save_plot:
        plt.savefig( save_plot , format='svg' , dpi=1200 )
    else:
        plt.savefig( save_plot )
        
        
        

# =============================================================================
# Plot composite ROC
# =============================================================================

def _plot_composite_roc_curve( trues , probs , n_arr  , cmet , file_in  ):
    from scipy import interp
    
    # Collect all TPR and FPR
    tpr_list  = []
    auc_list  = []
    sens_list = []
    spec_list = []
    
    max_length = 0

    mean_fpr = np.linspace(0, 1, 100)
    
    for i in range( n_arr ):
        # Get individual y_score and y_true
        y_score = np.array( probs[i] )
        y_true  = np.array( trues[i] )
                    
    
        # Compute metrics
        cmet._compute_metrics( y_true , y_score )

        tpr_list.append( interp( mean_fpr , cmet._fpr , cmet._tpr ) )
        tpr_list[-1][0] = 0.0
        auc_list.append( cmet._auc )
        sens_list.append( cmet._sensitivity )
        spec_list.append( cmet._specificity )
        
        if len( cmet._fpr ) > max_length:
            max_length = len( cmet._fpr )
            x_com      = cmet._fpr.copy()
        
        
    # Convert array to list    
    tpr_list  = np.array( tpr_list )    
    auc_list  = np.array( auc_list ) 
    sens_list = np.array( sens_list )    
    spec_list = np.array( spec_list )     
        
        
    # Construct the 3 curves
    mean_tpr     = np.mean( tpr_list , axis=0)
    mean_tpr[-1] = 1.0
    mean_auc     = me.auc( mean_fpr , mean_tpr )
    std_auc      = np.std( auc_list )

    std_tpr    = np.std( tpr_list , axis=0 )
    tprs_upper = np.minimum( mean_tpr + std_tpr , 1 )
    tprs_lower = np.maximum( mean_tpr - std_tpr , 0 )
 

    # Initialize figure
    fig = plt.figure()
    string = str( n_arr ) + '-fold CV'
    plt.title( string + ' composite ROC' , fontsize=16 ,fontweight='bold' )
    plt.xlabel( 'False Positive Rate' , fontsize=16 )
    plt.ylabel( 'True Positive Rate' , fontsize=16 )
    plt.xticks( fontsize=14 );  plt.yticks( fontsize=14 )


    # Plot "toss-coin" line y = x
    x_line = np.linspace( 0 , 1 , 50 )
    y_line = x_line.copy()
    plt.plot( x_line , y_line , '--' , lw=2 , color='red' )
        
        
    # Create additional text
    auc_mean  = np.mean( auc_list );  auc_std = np.std( auc_list )
    sens_mean = np.mean( sens_list );  sens_std = np.std( sens_list )
    spec_mean = np.mean( spec_list );  spec_std = np.std( spec_list )
    
    string = 'AUC  = ' + str( round( auc_mean , 2 ) ) + ' +/- ' + str( round( auc_std , 2 ) ) + '\n' + \
             'SENS = ' + str( round( sens_mean , 2 ) ) + ' +/- ' + str( round( sens_std , 2 ) ) + '\n' + \
             'SPEC = ' + str( round( spec_mean , 2 ) ) + ' +/- ' + str( round( spec_std , 2 ) )            
    
    plt.text( 0.60 , 
              0.20 , 
              string , 
              fontsize=13 , 
              horizontalalignment='left' ,
              verticalalignment='top' ,
              color='black' )       
        
        
    # Plot ROC curve
    plt.plot( mean_fpr , mean_tpr , lw=2 , color='black' , label='Mean ROC curve' )

    
    # Color area between min and max ROC
    plt.fill_between( mean_fpr , tprs_lower , tprs_upper , color='grey' , alpha=.35 ,
                      label=r'$\pm$ 1 std. dev.' )
          
    
    # Save figure
    plt.tight_layout()
    plt.legend( fontsize=10 )
    
    save_plot = file_in[:len( file_in )-5:] + '_composite_roc_curve.png'
        
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
    trues , probs , n_arr = _get_data_from_json( args.file_json )
    print( '\nNumber of arrays: ', n_arr )


    # Initialize metrics class
    cmet = Metrics( task      = 'class' ,
                    n_classes = 2       ,
                    metric    = 'auc'   ) 


    # Plot ROC curves
    print( '\nPlot separate ROC curve ....' )
    _plot_separate_roc_curves( trues , probs , n_arr , cmet , args.file_json )

    print( '\nPlot composite ROC curve ....' )
    _plot_composite_roc_curve( trues , probs , n_arr , cmet , args.file_json )

    print( '\n\n' )




# =============================================================================
# CALL TO MAIN
# =============================================================================

if __name__ == '__main__':
    main()

