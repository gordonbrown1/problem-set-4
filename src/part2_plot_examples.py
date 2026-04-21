'''
PART 2: PLOT EXAMPLES
- This code produces a range of standard plots using the seaborn library
- Walk through this code after you've run main.py to understand the plot types, syntax, and plots
- These are the types of plots you'll code in Parts 3, 4, 5
- NOTE: These aren't all the prettiest of plots. Pay better attention to color, formatting, and visual understandability in PARTS 3, 4, 5 and in general when making plots
'''

import seaborn as sns
import matplotlib.pyplot as plt

def seaborn_settings():
    '''
    Applies the default seaborn theme and sets the default figure size
    '''
    sns.set_theme()
    sns.set(rc={'figure.figsize':(6, 4)})


def barplots(charge_counts, charge_counts_by_offense):
    '''
    Produces various types of bar plots using the given datasets

    Parameters:
    - charge_counts dataframe
    - charge_counts_by_offense dataframe

    Returns:
    - Vertical barplot
    - Horizontal barplot
    - Vertical barplot with hue based on offense category
    '''
    sns.barplot(data=charge_counts, 
                x='charge_degree',
                y='count')
    plt.savefig('./data/part2_plots/vertical_barplot.png', bbox_inches='tight')

    sns.barplot(data=charge_counts, 
                y='charge_degree', 
                x='count', 
                orient='h')
    plt.savefig('./data/part2_plots/horizontal_barplot.png', bbox_inches='tight')

    sns.barplot(data=charge_counts_by_offense, 
                x='charge_degree',
                y='count',
                hue='offense_category')
    plt.savefig('./data/part2_plots/vertical_barplot_with_hue.png', bbox_inches='tight')


def cat_plots(charge_counts, pred_universe):
    '''
    Produces different types of categorical plots using the given datasets

    Parameters:
    - charge_counts dataframe
    - pred_universe dateframe

    Returns:
    - Categorical bar plot for charge degree counts
    - Categorical bar plot for non-felony predictions by sex
    '''
    sns.catplot(data=charge_counts,
                x='charge_degree',
                y='count', 
                kind='bar')
    plt.savefig('./data/part2_plots/catplot1.png', bbox_inches='tight')

    sns.catplot(data=pred_universe, 
                x='sex',
                y='prediction_nonfelony', 
                kind='bar')
    plt.savefig('./data/part2_plots/catplot2.png', bbox_inches='tight')


def histograms(pred_universe):
    '''
    Produces different types of histograms using the given dataset

    Parameters:
    - pred_universe dataframe

    Returns:
    - Histogram without specifying bins
    - Histogram with a specified number of bins
    - Histogram with specified bins and probability as the statistic
    '''
    sns.histplot(data=pred_universe, 
                 x='prediction_nonfelony')
    plt.savefig('./data/part2_plots/histogram1.png', bbox_inches='tight')

    sns.histplot(data=pred_universe, 
                 x='prediction_nonfelony',
                 bins=10)
    plt.savefig('./data/part2_plots/histogram2.png', bbox_inches='tight')

    sns.histplot(data=pred_universe, 
                 x='prediction_nonfelony', 
                 stat='probability',
                 bins=[0, .25, .8, 1])
    plt.savefig('./data/part2_plots/histogram3.png', bbox_inches='tight')


def scatterplot(pred_universe):
    '''
    Produces different types of scatter plots using the given dataset

    Parameters:
    - pred_universe dataframe

    Returns:
    - Scatterplot without a regression line
    - Scatterplot with a regression line
    - Scatterplot with a custom diagonal line
    - Scatterplot with hue by race
    - Scatterplot faceted by sex with hue by race
    '''
    sns.lmplot(data=pred_universe, 
               x='prediction_felony', 
               y='prediction_nonfelony',
               fit_reg=False)
    plt.savefig('./data/part2_plots/scatterplot1.png', bbox_inches='tight')

    sns.lmplot(data=pred_universe, 
               x='prediction_felony', 
               y='prediction_nonfelony')
    plt.savefig('./data/part2_plots/scatterplot2.png', bbox_inches='tight')

    sp = sns.lmplot(data=pred_universe, 
                    x='prediction_felony', 
                    y='prediction_nonfelony')
    sp.ax.axline(xy1=(0, 0), 
                 xy2=(1, 1),
                 color='g',
                 dashes=(2, 2))
    plt.savefig('./data/part2_plots/scatterplot3.png', bbox_inches='tight')

    sp = sns.lmplot(data=pred_universe, 
                    x='prediction_felony', 
                    y='prediction_nonfelony', 
                    hue='race')
    sp.ax.axline(xy1=(0, 0), 
                 xy2=(1, 1), 
                 color='b', 
                 dashes=(2, 2))
    plt.savefig('./data/part2_plots/scatterplot4.png', bbox_inches='tight')

    sp = sns.lmplot(data=pred_universe, 
                    x='prediction_felony', 
                    y='prediction_nonfelony', 
                    hue='race', 
                    col='sex')
    sp.axes[0][0].axline(xy1=(1, 1), slope=1, color='b', dashes=(2, 2))
    sp.axes[0][1].axline(xy1=(1, 1), slope=1, color='b', dashes=(2, 2))
    plt.savefig('./data/part2_plots/scatterplot5.png', bbox_inches='tight')
