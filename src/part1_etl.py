'''
PART 1: ETL
- This code sets up the datasets for Problem Set 4
- NOTE: You will update this code for PART 4: CATEGORICAL PLOTS
'''

import os
import pandas as pd

def create_directories(directories):
    """
    Creates the necessary directories for storing plots and data.

    Args:
        directories (list of str): A list of directory paths to create.
    """
    
    # Will create these ['data/part2_plots', 'data/part3_plots', 'data/part4_plots', 'data/part5_plots']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def extract_transform():
    """
    Extracts and transforms data from arrest records for analysis

    Returns:
        - `pred_universe`: The dataframe containing prediction-related data for individuals
        - `arrest_events`: The dataframe containing arrest event data
        - `charge_counts`: A dataframe with counts of charges aggregated by charge degree
        - `charge_counts_by_offense`: A dataframe with counts of charges aggregated by both charge degree and offense category
    """
    # Extracts arrest data CSVs into dataframes
    pred_universe = pd.read_csv('https://www.dropbox.com/scl/fi/a2tpqpvkdc8n6advvkpt7/universe_lab9.csv?rlkey=839vsc25njgfftzakr34w2070&dl=1')
    arrest_events = pd.read_csv('https://www.dropbox.com/scl/fi/n47jt4va049gh2o4bysjm/arrest_events_lab9.csv?rlkey=u66usya2xjgf8gk2acq7afk7m&dl=1')

    # Creates two additional dataframes using groupbys
    charge_counts = arrest_events.groupby(['charge_degree']).size().reset_index(name='count')
    charge_counts_by_offense = arrest_events.groupby(['charge_degree', 'offense_category']).size().reset_index(name='count')
    
    return pred_universe, arrest_events, charge_counts, charge_counts_by_offense


def create_felony_charge_dataframe(arrest_events):
    """
    Creates a dataframe indicating whether each arrest had at least one felony charge.
    
    Parameters:
    arrest_events : pandas.DataFrame
        The arrest events dataframe containing 'arrest_id' and 'charge_degree' columns
        
    Returns:
    pandas.DataFrame
        A dataframe with columns ['arrest_id', 'has_felony_charge'] where has_felony_charge
        is a boolean indicating if the arrest included at least one felony charge.
    """
    # Group by arrest_id and check if any charge_degree is 'felony'
    felony_charge = arrest_events.groupby('arrest_id').agg(
    has_felony_charge=('charge_degree', lambda x: (x == 'felony').any())).reset_index()
    return felony_charge

# 2. Merge `felony_charge` with `pre_universe` into a new dataframe
def merge_felony_with_universe(pred_universe, felony_charge):
    """
    Merges the felony charge indicator with the prediction universe dataframe.
    
    Parameters:
    pred_universe : pandas.DataFrame
        The prediction universe dataframe
    felony_charge : pandas.DataFrame
        The felony charge indicator dataframe with 'arrest_id' and 'has_felony_charge' columns
        
    Returns:
    pandas.DataFrame
        The merged dataframe containing all original columns plus 'has_felony_charge'
    """
    merged_df = pred_universe.merge(felony_charge, on="arrest_id", how="left")
    return merged_df
