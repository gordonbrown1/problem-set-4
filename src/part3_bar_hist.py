'''
PART 3: BAR PLOTS AND HISTOGRAMS
- Write functions for the tasks below
- Update main() in main.py to generate the plots and print statments when called
- All plots should be output as PNG files to `data/part3_plots`
'''
import seaborn as sns
import matplotlib.pyplot as plt


# 1. Using the pre_universe data frame, create a bar plot for the fta column.
def plot_fta_bar(pred_universe):
    '''
    Creates a bar plot showing the distribution of Failure to Appear (FTA) status.
    
    Parameters:
    pred_universe : pandas.DataFrame
        The prediction universe dataframe containing the 'fta' column
        
    Returns:
    None
        Saves plot to './data/part3_plots/fta_barplot.png'
    '''
    # Create bar plot counting occurrences of each FTA value
    sns.countplot(data=pred_universe, x="fta")

    # Adding labels and titles
    plt.title("Failure to Appear Distribution")
    plt.xlabel("FTA Stauts")
    plt.ylabel("Counts")
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Save plot
    plt.savefig("./data/part3_plots/fta_barplot.png", bbox_inches="tight")
    plt.close()





# 2. Hue the previous barplot by sex
def plot_fta_bar_hued(pred_universe):
    '''
    Creates a bar plot showing the distribution of Failure to Appear (FTA) status,
    grouped by sex.
    
    Parameters:
    pred_universe : pandas.DataFrame
        The prediction universe dataframe containing 'fta' and 'sex' columns
        
    Returns:
    None
        Saves plot to './data/part3_plots/fta_barplot_hued.png'
    '''
    # Create count plot with hue for sex
    sns.countplot(data=pred_universe,x='fta',hue='sex')
    
    # Add labels and title
    plt.title('Failure to Appear (FTA) Distribution by Sex')
    plt.xlabel('FTA Status')
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Save and close the figure
    plt.savefig('./data/part3_plots/fta_barplot_hued.png', bbox_inches='tight')
    plt.close()



# 3. Plot a histogram of age_at_arrest
def plot_age_histogram(pred_universe):
    '''
    Creates a histogram showing the distribution of age at arrest.
    
    Parameters:
    pred_universe : pandas.DataFrame
        The prediction universe dataframe containing the 'age_at_arrest' column
        
    Returns:
    None
        Saves plot to './data/part3_plots/age_histogram.png'
    '''
    # Create histogram with default binning
    sns.histplot(
        data=pred_universe,
        x='age_at_arrest')
    
    # Add labels and title
    plt.title('Distribution of Age at Arrest')
    plt.xlabel('Age at Arrest')
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Save and close the figure
    plt.savefig('./data/part3_plots/age_histogram.png', bbox_inches='tight')
    plt.close()


# 4. Plot the same histogram, but create bins that represent the following age groups 
#  - 18 to 21
#  - 21 to 30
#  - 30 to 40
#  - 40 to 100 

def plot_age_histogram_binned(pred_universe):
    '''
    Creates a histogram showing the distribution of age at arrest using specified
    age group bins
    
    Parameters:
    pred_universe : pandas.DataFrame
        The prediction universe dataframe containing the 'age_at_arrest' column
        
    Returns:
    None
        Saves plot to './data/part3_plots/age_histogram_binned.png'
    '''
    # Define age group boundaries
    bins = [18, 21, 30, 40, 100]
    
    # Create histogram with custom bins
    sns.histplot(data=pred_universe, x='age_at_arrest', bins=bins)
    
    # Add labels and title
    plt.title('Distribution of Age at Arrest (Grouped by Age Ranges)')
    plt.xlabel('Age at Arrest')
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Save and close the figure
    plt.savefig('./data/part3_plots/age_histogram_binned.png', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    '''
    Simple test section for direct execution of this file.
    Loads data using the ETL function from part1 and generates all Part 3 plots.
    '''
    print("Testing part3_bar_hist.py directly...")
    
    # Import the ETL function from part1
    from part1_etl import extract_transform
    
    # Load the data
    print("Loading data...")
    pred_universe, arrest_events, charge_counts, charge_counts_by_offense = extract_transform()
    print("Data loaded successfully.")
    
    # Test all plotting functions
    print("\nGenerating plot 1: FTA bar plot...")
    plot_fta_bar(pred_universe)

    print("\nGenerating plot 2: FTA bar plot with sex hue...")
    plot_fta_bar_hued(pred_universe)
    
    print("Generating plot 3: Age histogram...")
    plot_age_histogram(pred_universe)

    print("Generating plot 4: Age histogram with bins...")
    plot_age_histogram_binned(pred_universe)
