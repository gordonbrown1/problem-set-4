'''
PART 4: CATEGORICAL PLOTS
- Write functions for the tasks below
- Update main() in main.py to generate the plots and print statments when called
- All plots should be output as PNG files to `data/part4_plots`
'''

import seaborn as sns
import matplotlib.pyplot as plt

##  UPDATE `part1_etl.py`  ##
# 1. The charge_no column in arrest events tells us the charge degree and offense category for each arrest charge. 
# An arrest can have multiple charges. We want to know if an arrest had at least one felony charge.
# Use groupby and apply with lambda to create a new dataframe called `felony_charge` that has columns: ['arrest_id', 'has_felony_charge']
# 
# Hint 1: One way to do this is that in the lambda function, check to see if a charge_degree is felony, sum these up, and then check if the sum is greater than zero. 
# Hint 2: Another way to do thisis that in the lambda function, use the `any` function when checking to see if any of the charges in the arrest are a felony



# 2. Merge `felony_charge` with `pre_universe` into a new dataframe

# 3. You will need to update ## PART 1: ETL ## in main() to call these two additional dataframes


##  PLOTS  ##
# 1. Create a catplot where the categories are charge type and the y-axis is the prediction for felony rearrest. Set kind='bar'.
def plot_felony_prediction_by_charge_type(merged_df):
    '''
    Creates a catplot showing felony rearrest predictions grouped by current charge type.
    
    Parameters:
    merged_df : pandas.DataFrame
        The merged dataframe containing prediction data and felony charge indicators
        
    Returns:
    None
        Saves plot to './data/part4_plots/felony_prediction_by_charge_type.png'
    '''
    sns.catplot(
        data=merged_df,x='has_felony_charge',y='prediction_felony',kind='bar')
    
    plt.title('Felony Rearrest Prediction by Charge Type')
    plt.xlabel('Charge is Felony')
    plt.ylabel('Predicted Probability of Felony Rearrest')
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    
    plt.savefig('./data/part4_plots/felony_prediction_by_charge_type.png', bbox_inches='tight')
    plt.close()

# 2. Now repeat but have the y-axis be prediction for nonfelony rearrest
def plot_nonfelony_prediction_by_charge_type(merged_df):
    '''
    Creates a catplot showing nonfelony rearrest predictions grouped by current charge type.
    
    Parameters:
    merged_df : pandas.DataFrame
        The merged dataframe containing prediction data and felony charge indicators
        
    Returns:
    None
        Saves plot to './data/part4_plots/nonfelony_prediction_by_charge_type.png'
    '''
    sns.catplot(
        data=merged_df,x='has_felony_charge',y='prediction_nonfelony',kind='bar')
    
    plt.title('Nonfelony Rearrest Prediction by Charge Type')
    plt.xlabel('Charge is Felony')
    plt.ylabel('Predicted Probability of Nonfelony Rearrest')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.savefig('./data/part4_plots/nonfelony_prediction_by_charge_type.png', bbox_inches='tight')
    plt.close()
    
 # In a print statement, answer the following question: What might explain the difference between the plots?
    print("\nPART 4 - Question 2:")
    print("What might explain the difference between the plots?")
    print("-" * 70)
    print("The model identifies individuals with at least a felony current charge history")
    print("as a high risk of rearrest on any charge (misdemeanor/felony). Also, individual")
    print("with no felony history have a high predicted probability for nonfelony charge")
    print("indicating that although a person might not have prior history of felony doesn't")
    print("rule the fact that they could be rearrested for other nonfelonies")
    print("\nIndividuals with history of felony have relatively lower avg. probability of being charged with felony again,")
    print("but high in nonfelony, this suggest the model is training on other predictors with different classes,")
    print("which is independent on prior charge degree\nlike: ")
    print("    sex: M and F")
    print("    race: A and B")
    print("    Or maybe individuals do not repeat felonies\n")



# 3. Repeat the plot from 1, but hue by whether the person actually got rearrested for a felony crime
# 
# In a print statement, answer the following question: 
# What does it mean that prediction for arrestees with a current felony charge, 
# but who did not get rearrested for a felony crime have a higher predicted probability than arrestees with a current misdemeanor charge, 
# but who did get rearrested for a felony crime?
def plot_felony_prediction_hued_by_actual_rearrest(merged_df):
    '''
    Creates a catplot showing felony rearrest predictions grouped by current charge type
    and hued by whether the person actually got rearrested for a felony.
    
    Parameters:
    merged_df : pandas.DataFrame
        The merged dataframe containing prediction data, felony charge indicators, 
        and actual rearrest outcomes
        
    Returns:
    None
        Saves plot to './data/part4_plots/felony_prediction_hued_by_rearrest.png'
    '''
    # Create a copy to avoid modifying original dataframe
    plot_df = merged_df.copy()
    
    # Create readable labels for x-axis
    plot_df['charge_type'] = plot_df['has_felony_charge'].map({
        True: 'Felony Charge',
        False: 'Misdemeanor Charge'})
    
    # Create readable labels for the legend
    plot_df['rearrest_status'] = plot_df['y_felony'].map({
        1.0: 'Rearrested for Felony',
        0.0: 'Not Rearrested for Felony'})
    
    # Set a clean style
    sns.set_style("whitegrid")
    
    # Create the plot with custom color palette
    g = sns.catplot(
        data=plot_df,
        x='charge_type',
        y='prediction_felony',
        hue='rearrest_status',
        kind='bar',
        palette='Set2',
        height=5,
        aspect=1.2,
        legend=True)
    
    # Customize the plot appearance
    g.ax.set_title('Felony Rearrest Prediction by Current Charge Type and Actual Outcome', 
                   fontsize=12, fontweight='bold', pad=15)
    g.ax.set_xlabel('Current Charge Type', fontsize=10, fontweight='bold')
    g.ax.set_ylabel('Predicted Probability of Felony Rearrest', fontsize=10, fontweight='bold')
    
    # Access legend through the figure-level object and modify it
    if g.legend is not None:
        g.legend.set_title('Actual Outcome')
        g.legend.set_loc('upper left')
        g.legend.get_title().set_fontsize(10)
        for text in g.legend.get_texts():
            text.set_fontsize(9)
    
    # Add gridlines for better readability
    g.ax.grid(True, linestyle='--', alpha=0.6)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('./data/part4_plots/felony_prediction_hued_by_rearrest.png', 
                bbox_inches='tight', dpi=150)
    plt.close()
    
    # Print analysis
    print("\nPART 4 - Question 3:")
    print("-" * 60)

    print("It means the model gives more weight to an individual's current charge type.")
    print("A person with a current felony charge has higher risk for future felony rearrest,")
    print("even if that person does not actually end up being rearrested for a felony.")
    print("At the same time, a person with a current misdemeanor charge may receive a lower predicted risk,")
    print("even if that person actually does later get rearrested for a felony.")
    print("The model is relying strongly on current felony charge as a risk flag.\n")
    

if __name__ == "__main__":
    '''
    Simple test section for direct execution of this file.
    '''
    print("Testing part4_catplot.py directly...")
    
    # Import functions from part1_etl
    from part1_etl import (
        extract_transform, 
        create_felony_charge_dataframe, 
        merge_felony_with_universe,
        create_directories)
    
    print("Creating directories...")
    directories = ['data/part4_plots']
    create_directories(directories)
    
    print("Loading data...")
    pred_universe, arrest_events, charge_counts, charge_counts_by_offense = extract_transform()
    
    print("Creating felony charge dataframe...")
    felony_charge = create_felony_charge_dataframe(arrest_events)
    
    print("Merging dataframes...")
    merged_df = merge_felony_with_universe(pred_universe, felony_charge)
    
    print("Generating plots...")
    plot_felony_prediction_by_charge_type(merged_df)
    plot_nonfelony_prediction_by_charge_type(merged_df)
    plot_felony_prediction_hued_by_actual_rearrest(merged_df)