'''
- You will run Problem Set 4 from this .py, so make sure to set things up to return outputs accordingly
- Go through each PART and write code / make updates as necessary to produce all required outputs
- Run main.py before you start
'''

import warnings
import numpy as np

# Suppress numpy linear algebra warnings
warnings.filterwarnings('ignore')
np.seterr(divide='ignore', invalid='ignore')

import src.part1_etl as part1
import src.part2_plot_examples as part2
import src.part3_bar_hist as part3
import src.part4_catplot as part4
import src.part5_scatter as part5

def main():
    ##  PART 1: ETL  ##
    # ETL the datasets into dataframes
    directories = ['data/part2_plots', 'data/part3_plots', 'data/part4_plots', 'data/part5_plots']
    part1.create_directories(directories)
    
    pred_universe, arrest_events, charge_counts, charge_counts_by_offense = part1.extract_transform()
    
   ##  PART 2: PLOT EXAMPLES  ##
    # Apply plot theme
    part2.seaborn_settings()

    # Generate plots
    part2.barplots(charge_counts, charge_counts_by_offense)
    part2.cat_plots(charge_counts, pred_universe)
    part2.histograms(pred_universe)
    part2.scatterplot(pred_universe)

    ##  PART 3: BAR PLOTS AND HISTOGRAMS  ##
    # 1
    part3.plot_fta_bar(pred_universe)

    # 2
    part3.plot_fta_bar_hued(pred_universe)

    # 3
    part3.plot_age_histogram(pred_universe)

    # 4
    part3.plot_age_histogram_binned(pred_universe)


    ## PART 4: CATEGORICAL PLOTS  ##
  
    felony_charge = part1.create_felony_charge_dataframe(arrest_events)
    merged_df = part1.merge_felony_with_universe(pred_universe, felony_charge)

    # 1. Create catplot for felony rearrest prediction by charge type
    part4.plot_felony_prediction_by_charge_type(merged_df)
    
    # 2. Create catplot for nonfelony rearrest prediction by charge type
    part4.plot_nonfelony_prediction_by_charge_type(merged_df)
    
    # 3. Create catplot hued by actual felony rearrest
    part4.plot_felony_prediction_hued_by_actual_rearrest(merged_df)

    ##  PART 5: SCATTERPLOTS  ##
    # 1
    part5.plot_predictions_scatter_hued_by_charge(merged_df)

    # 2
    part5.plot_calibration_scatter(merged_df)


if __name__ == "__main__":
    main()
