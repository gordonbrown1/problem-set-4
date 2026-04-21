'''
PART 5: SCATTER PLOTS
- Write functions for the tasks below
- Update main() in main.py to generate the plots and print statments when called
- All plots should be output as PNG files to `data/part5_plots`
'''

import seaborn as sns
import matplotlib.pyplot as plt

# 1. Using lmplot, create a scatter plot where the x-axis is the prediction for felony and the y-axis the is prediction for a nonfelony, and hue this by whether the current charge is a felony. 
# 
# In a print statement, answer the following question: What can you say about the group of dots on the right side of the plot?
def plot_predictions_scatter_hued_by_charge(merged_df):
    '''
    Creates a scatter plot showing felony vs nonfelony predictions, 
    hued by whether the current charge is a felony.
    
    Parameters:
    merged_df : pandas.DataFrame
        The merged dataframe containing prediction data and has_felony_charge column
        
    Returns:
    None
        Saves plot to './data/part5_plots/predictions_scatter_by_charge.png'
    '''
    # Create a copy to avoid modifying original dataframe
    plot_df = merged_df.copy()
    
    # Create readable labels for the hue
    plot_df['charge_type'] = plot_df['has_felony_charge'].map({
        True: 'Felony Charge',
        False: 'Misdemeanor Charge'})
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create lmplot without regression line
    g = sns.lmplot(
        data=plot_df,x='prediction_felony',
        y='prediction_nonfelony',hue='charge_type',palette='Set1',height=6,
        aspect=1.2, fit_reg=False,scatter_kws={'alpha': 0.5, 's': 20})
    
    # Add a positive 45-degree diagonal reference line (y = x)
    for ax in g.axes.flat:
        ax.axline(xy1=(0, 0), xy2=(1, 1), color='gray', linestyle='--', 
                  alpha=0.5, linewidth=1.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    # Customize plot appearance
    g.ax.set_title('Felony vs Nonfelony Predictions by Current Charge Type', 
                   fontsize=12, fontweight='bold', pad=15)
    g.ax.set_xlabel('Predicted Probability of Felony Rearrest', 
                    fontsize=10, fontweight='bold')
    g.ax.set_ylabel('Predicted Probability of Nonfelony Rearrest', 
                    fontsize=10, fontweight='bold')
    
    # Modify legend position and styling
    if g.legend is not None:
        g.legend.set_title('Current Charge Type')
        g.legend.set_loc('upper right')
        g.legend.get_title().set_fontsize(10)
        for text in g.legend.get_texts():
            text.set_fontsize(9)
    
    plt.tight_layout()
    plt.savefig('./data/part5_plots/predictions_scatter_by_charge.png', 
                bbox_inches='tight', dpi=150)
    plt.close()
    print("PART 5 - Question 1:")
    print("What can you say about the group of dots on the right side of the plot?")
    print("-" * 70)
    print("The group of dots on the right side mostly represents people with a current felony charge.")
    print("They have relatively high predicted probabilities of felony rearrest and also fairly high predicted \nprobabilities of nonfelony rearrest.")
    print("This suggests the model is identifying this group as high overall rearrest risk.")
    print("Many of those dots are still above the diagonal line, which means their predicted probability \nof nonfelony rearrest is often higher than their predicted probability of felony rearrest.\n")
    

# 2. Create a scatterplot where the x-axis is prediction for felony rearrest and the y-axis is whether someone was actually rearrested.
# 
# In a print statement, answer the following question: Would you say based off of this plot if the model is calibrated or not?
def plot_calibration_scatter(merged_df):
    '''
    Creates a scatter plot showing predicted felony rearrest probability vs 
    actual felony rearrest outcome to assess model calibration.
    
    Parameters:
    merged_df : pandas.DataFrame
        The merged dataframe containing prediction_felony and y_felony columns
        
    Returns:
    None
        Saves plot to './data/part5_plots/calibration_scatter.png'
    '''
    # Set style
    sns.set_style("whitegrid")
    
    # Create lmplot with logistic regression (doesn't require statsmodels)
    g = sns.lmplot(
        data=merged_df,
        x='prediction_felony',
        y='y_felony',
        logistic=True,
        height=6,
        aspect=1.2,
        scatter_kws={'alpha': 0.3, 's': 15},
        line_kws={'color': 'red', 'linewidth': 2})
    
    # Add ideal calibration line 
    for ax in g.axes.flat:
        ax.axline(xy1=(0, 0), xy2=(1, 1), color='blue', linestyle='--', 
                  alpha=0.7, linewidth=1.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.05)
    
    # Customize plot appearance
    g.ax.set_title('Model Calibration: Predicted vs Actual Felony Rearrest', 
                   fontsize=12, fontweight='bold', pad=15)
    g.ax.set_xlabel('Predicted Probability of Felony Rearrest', 
                    fontsize=10, fontweight='bold')
    g.ax.set_ylabel('Actual Felony Rearrest (0 = No, 1 = Yes)', 
                    fontsize=10, fontweight='bold')
    
    # Add custom legend
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='red', linewidth=2, label='Logistic Trend (Observed)'),
        Line2D([0], [0], color='blue', linestyle='--', linewidth=1.5, 
               label='Perfect Calibration (y = x)')]
    g.ax.legend(handles=custom_lines, loc='upper left', bbox_to_anchor=(0.03, 0.92), fontsize=9)
    
    plt.tight_layout()
    plt.savefig('./data/part5_plots/calibration_scatter.png', 
                bbox_inches='tight', dpi=150)
    plt.close()

    print("PART 5 - Question 2:")
    print("Would you say based off of this plot if the model is calibrated or not?")
    print("-" * 70)

    print("The model is not perfectly calibrated.")
    print("There are many cases with low predicted probability that were actually rearrested for a felony.")
    print("This means the model is underestimating high risk individual who truly had the outcome.")
    print("The model finds it difficiult seperating certain class with higher risk of felony rearrest\nranking them lower and vice-versa.")
    print("Individual points show that the model still misses many actual felony rearrests at low \npredicted probabilities.\n")


if __name__ == "__main__":
    '''
    Simple test section for direct execution of this file.
    '''
    print("Testing part5_scatter.py directly...")
    
    # Import functions from part1_etl
    from part1_etl import (
        extract_transform, 
        create_felony_charge_dataframe, 
        merge_felony_with_universe,
        create_directories)
    
    print("Creating directories...")
    directories = ['data/part5_plots']
    create_directories(directories)
    
    print("Loading data...")
    pred_universe, arrest_events, charge_counts, charge_counts_by_offense = extract_transform()
    
    print("Creating felony charge dataframe...")
    felony_charge = create_felony_charge_dataframe(arrest_events)
    
    print("Merging dataframes...")
    merged_df = merge_felony_with_universe(pred_universe, felony_charge)
    
    print("Data loaded successfully.")
    print(f"Total records: {len(merged_df)}")
    print()
    
    print("Generating plots...")
    plot_predictions_scatter_hued_by_charge(merged_df)
    plot_calibration_scatter(merged_df)
    
    