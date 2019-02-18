'''
SYMBOLIC MODELING TRAINING                                                                                   
'''

  
# Author   : Filippo Arcadu
# Date     : 10/11/2018
 
 
from __future__ import division , print_function
import sys , os
import argparse
import time


from syml_class import SYML




# =============================================================================
# Get input arguments 
# =============================================================================

def _example():
    print( '\n\nTrain model given a CSV table and a YAML config file:\n' )
    print( '"""\npython syml_train.py -i data.csv -c syml_config.yml -o ./\n"""\n\n' )


def _get_args():
    parser = argparse.ArgumentParser( description     = 'Training Symbolic Machine Learning Model',
                                      formatter_class = argparse.ArgumentDefaultsHelpFormatter    ,
                                      add_help        = False )
    
    parser.add_argument('-i', '--file_in', dest='file_in',
                        help='Specify input CSV file containing features and output variable')

    parser.add_argument('-c', '--config', dest='file_cfg',
                        help='Specify destination where to save outputs')  
    
    parser.add_argument('-o', '--path_out', dest='path_out', default='./' ,
                        help='Specify destination where to save outputs')  
    
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                        help='Enable verbose mode' )
    
    parser.add_argument('-a', '--label', dest='label',
                        help='Add label to output files' )
    
    parser.add_argument('-p', '--print_vars', dest='print_vars', action='store_true',
                        help='Print names and unique values of features and outcome variables' )

    parser.add_argument('-h', '--help', dest='help', action='store_true',
                        help='Print example command line' )

    args = parser.parse_args()

    if args.help:
        parser.print_help()
        _example()
        sys.exit()

    if args.file_in is None:
        parser.print_help()
        sys.exit('\nERROR: Input CSV file not specified!\n')

    if os.path.isfile( args.file_in ) is False:
        parser.print_help()
        sys.exit('\nERROR: Input CSV file does not exist!\n')

    if args.file_cfg is None:
        parser.print_help()
        sys.exit('\nERROR: Input YAML file not specified!\n')

    if os.path.isfile( args.file_cfg ) is False:
        parser.print_help()
        sys.exit('\nERROR: Input YAML file does not exist!\n')

    return args




# =============================================================================
# MAIN 
# =============================================================================

def main():
    # Initial print
    time1 = time.time()
    
    print( '\n\n' )
    print( '#########################################' )
    print( '#                                       #' )
    print( '#        TRAIN SYMBOLIC ML MODEL        #' )
    print( '#                                       #' )
    print( '#########################################' )
    print( '\n' )


    # Get input arguments
    args = _get_args()


    # Initialize SYML class
    syml = SYML( args.file_in             ,
                 args.file_cfg            ,
                 path_out = args.path_out ,
                 label    = args.label    , 
                 verbose  = args.verbose  )


    # Check prints
    print( '\nINPUTS' )
    print( 'CSV data file: ', syml._file_in )
    print( 'Shape of input data frame: ', syml._df.shape )
    print( 'YAML config file: ', syml._file_cfg )
    print( 'Additional label: ', syml._add_label )
    
    print( '\nMODEL' )
    print( 'Algorithm: ', syml._alg )
    print( 'Metric: ', syml._metric )

    print( '\nFEATURES' )
    print( 'Output variable: ', syml._col_out )
    
    if syml._select is not None:
        print( 'Feature selection: ', syml._select )

    if syml._exclude is not None:
        print( 'Excluded features: ', syml._feats_excl )

    print( 'Number of features for modeling: ', len( syml._feats ) )
    print( '\nName of features for modeling:\n' , syml._feats )
    print( '\nName of categorical features for modeling:\n' , syml._feats_cat )

    print( '\nVALIDATION' )
    print( 'Constrained feature for splitting: ', syml._col_constr )
    
    if syml._cv_type == 'k-fold': 
        print( 'K-fold cross-validation' )
        print( 'No. folds for cross-validation: ', syml._n_splits )
    elif syml._cv_type == 'resampling':
        print( 'Random re-sampling cross-validation' )
        print( 'Percentage of data for testing: ', syml._test_perc )
        print( 'No. splits for cross-validation: ', syml._n_splits )
    
    print( 'Label for output files: ', syml._label_out )

    print( '\nPARAMETER GRID' )
    print( 'Number of parameter grid points: ', len( syml._param_combs ) )
    print( 'Input parameters: ', syml._params )

    
    # Check feature and outcome variables
    if args.print_vars:
        print( '\n\nChecking feature and outcome variables ....' )
        syml._check_vars()


    # Training
    syml._train()


    # Elapsed time
    time2 = time.time()
    print( '\n\nTime elapsed: ', time2-time1 )
    print( '\n\n' )




# =============================================================================
# CALL to MAIN 
# =============================================================================

if __name__ == '__main__':
    main()
