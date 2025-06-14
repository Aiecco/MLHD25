# src/plot/plot_evaluation.py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def plot_eval(errors_list, mae_months, true_months_list, pred_months_list):
    """
        Generates and saves a series of plots to analyze the performance of
        the age prediction model.

        Args:
            errors_list (list/array): List of absolute prediction errors (months).
            mae_months (float): Overall Mean Absolute Error (months).
            true_months_list (list/array): List of true ages (months).
            pred_months_list (list/array): List of predicted ages (months).
        """
    plt.figure(figsize=(18, 12))

    # 1. Distribution of absolute errors
    plt.subplot(2, 2, 1)
    sns.histplot(errors_list, kde=True, bins=30)
    plt.axvline(mae_months, color='r', linestyle='--', label=f'MAE: {mae_months:.2f} months')
    plt.xlabel('Absolute Error (months)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Age Prediction Errors')
    plt.legend()

    # 2. Scatter plot: true age vs predicted age
    plt.subplot(2, 2, 2)
    plt.scatter(true_months_list, pred_months_list, alpha=0.5)
    # Ideal y=x line
    min_val = min(min(true_months_list), min(pred_months_list))
    max_val = max(max(true_months_list), max(pred_months_list))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
    plt.xlabel('True Age (months)')
    plt.ylabel('Predicted Age (months)')
    plt.title('True Age vs Predicted Age')
    plt.legend()

    # 3. Box plot of errors by age group
    plt.subplot(2, 2, 3)

    # Base age bins in months up to 20 years
    # These represent the lower bounds of the intervals for standard labels
    standard_age_boundaries_months = [0, 36, 72, 120, 180]  # 0, 3, 6, 10, 15 years
    standard_age_labels = ['0-3', '3-6', '6-10', '10-15']  # Labels for these intervals

    # Calculate the maximum true age in months from the data
    max_true_age_in_months = max(true_months_list)

    current_age_bins = list(standard_age_boundaries_months)
    current_age_labels = list(standard_age_labels)

    # Determine the upper bound for the last interval
    # The last standard boundary is 180 months (15 years)
    # We want the next bin to start at 15 years (180 months) and end at max_true_age,
    # or at 20 years if max_true_age is less than 20.

    # Calculate the upper limit for the final bin, ensuring it covers the max age
    # It should be at least 20 years (240 months) if data goes up to there,
    # or the max_true_age + a small margin to ensure inclusion with right=False.
    final_bin_upper_bound = max(240, int(max_true_age_in_months) + 1)  # Ensure it goes at least to 20 years + 1 month
    # and covers max data age

    if final_bin_upper_bound > current_age_bins[-1]:  # If max age is > 15 years (180 months)
        current_age_bins.append(final_bin_upper_bound)

        # Determine the start year for the last label (e.g., 15 for 15-X)
        start_year_for_last_bin = int(current_age_bins[-2] / 12)
        # Determine the end year for the last label (round up to nearest year for clarity)
        end_year_for_last_bin = int(np.ceil(max_true_age_in_months / 12.0))

        # Ensure the end year is at least 20 if the data goes up to 20 years
        if max_true_age_in_months >= 240:  # If data includes 20 years or more
            end_year_for_last_bin = max(20, end_year_for_last_bin)  # Ensure it's at least 20

        # Create the label for the last bin
        current_age_labels.append(f'{start_year_for_last_bin}-{end_year_for_last_bin}')

    # Create a pandas Series for age
    age_series = pd.Series(true_months_list)
    # Using 'right=False' for [a, b) intervals.
    binned_ages = pd.cut(age_series, bins=current_age_bins, labels=current_age_labels, right=False)

    # Create a DataFrame with age and errors
    error_df = pd.DataFrame({
        'Age_Group': binned_ages,
        'Absolute_Error': errors_list
    })

    # Filter out NaNs if any, which might occur if data falls outside of the determined bins
    error_df = error_df.dropna(subset=['Age_Group'])

    # Ensure all labels appear in the correct order, even if some bins are empty
    error_df['Age_Group'] = pd.Categorical(error_df['Age_Group'], categories=current_age_labels, ordered=True)

    # Box plot
    sns.boxplot(x='Age_Group', y='Absolute_Error', data=error_df)
    plt.xlabel('Age Group (years)')
    plt.ylabel('Absolute Error (months)')
    plt.title('Distribution of Errors by Age Group')
    plt.xticks(rotation=45)

    # 4. Heatmap: error based on true vs predicted age
    plt.subplot(2, 2, 4)

    # Convert months to years for readability
    true_years_array = np.array(true_months_list) / 12
    pred_years_array = np.array(pred_months_list) / 12

    # Define base bins for years (e.g., 0-4, 4-8, 8-12, 12-16, 16-20)
    base_bins_years = np.arange(0, 21, 4)  # Up to 20 years (20 is the upper bound of the last base bin)

    # Determine the effective upper bound for bins based on actual max data
    max_data_year_heatmap = max(np.max(true_years_array), np.max(pred_years_array))

    # Ensure the last bin in bins_years covers up to max_data_year without a '+' label
    current_bins_years = list(base_bins_years)

    if max_data_year_heatmap > base_bins_years[-1]:
        # Extend the last bin's upper limit to include max_data_year
        # Add 1 to ensure pd.histogram2d includes values up to max_data_year (similar to right=False in pd.cut)
        current_bins_years.append(max_data_year_heatmap + 1)

    # Calculate 2D distributions with np.histogram2d
    heatmap, xedges, yedges = np.histogram2d(true_years_array, pred_years_array,
                                             bins=[current_bins_years, current_bins_years])

    # Normalize by row to show the distribution of predictions for each true age group
    row_sums = heatmap.sum(axis=1, keepdims=True)
    heatmap_norm = np.divide(heatmap, row_sums, out=np.zeros_like(heatmap), where=row_sums != 0)

    # Create labels for the heatmap axes dynamically, ensuring no '+' sign
    x_labels = []
    y_labels = []

    for i in range(len(xedges) - 1):
        lower_bound = int(xedges[i])
        upper_bound = int(xedges[i + 1])
        # For the last bin, if it was extended, label it with the actual range
        if i == len(xedges) - 2 and upper_bound > base_bins_years[-1]:
            x_labels.append(f'{lower_bound}-{int(np.floor(max_data_year_heatmap))}')
        else:
            x_labels.append(f'{lower_bound}-{upper_bound}')

    for i in range(len(yedges) - 1):
        lower_bound = int(yedges[i])
        upper_bound = int(yedges[i + 1])
        if i == len(yedges) - 2 and upper_bound > base_bins_years[-1]:
            y_labels.append(f'{lower_bound}-{int(np.floor(max_data_year_heatmap))}')
        else:
            y_labels.append(f'{lower_bound}-{upper_bound}')

    sns.heatmap(heatmap_norm, cmap='YlGnBu', annot=True, fmt=".2f",
                xticklabels=x_labels,
                yticklabels=y_labels)
    plt.xlabel('Predicted Age (years)')
    plt.ylabel('True Age (years)')
    plt.title('Distribution of Predictions by Age Group (Row Normalized)')

    plt.tight_layout()
    plt.savefig('age_prediction_analysis.png', dpi=300)
    plt.show()