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

    # Create age bins
    # Extended bins to cover a wider range if necessary
    age_bins = [0, 36, 72, 120, 180, 240, 300]  # Years: 0-3, 3-6, 6-10, 10-15, 15-20, 20-25
    age_labels = ['0-3', '3-6', '6-10', '10-15', '15-20', '20+']  # Corresponding to the bins

    # Ensure age_bins covers the entire age range of your data
    if max(true_months_list) > age_bins[-1]:
        # Add a final bin if there are ages beyond the defined maximum
        age_bins.append(int(max(true_months_list) + 12))  # Add one more year to the max
        age_labels.append(f'{age_labels[-1]}+')  # Update the label for the last bin

    # Create a pandas Series for age
    age_series = pd.Series(true_months_list)
    binned_ages = pd.cut(age_series, bins=age_bins, labels=age_labels, right=False)  # right=False for interval [a, b)

    # Create a DataFrame with age and errors
    error_df = pd.DataFrame({
        'Age_Group': binned_ages,
        'Absolute_Error': errors_list
    })

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

    # Define bins for years (e.g., 0-4, 4-8, 8-12, 12-16, 16-20, 20+)
    # Use a step for the bins and ensure they cover the entire age range
    max_age_in_years = max(np.max(true_years_array), np.max(pred_years_array))
    # Create dynamic bins that include all existing ages
    # Example: bins every 4 years, but the last bin includes the maximum
    bins_years = np.arange(0, max_age_in_years + 5, 4)  # Add a little margin
    if max_age_in_years > bins_years[-1]:  # Ensure the last bin covers the maximum
        bins_years = np.append(bins_years, max_age_in_years + 1)

    # Calculate 2D distributions with np.histogram2d
    heatmap, xedges, yedges = np.histogram2d(true_years_array, pred_years_array, bins=[bins_years, bins_years])

    # Normalize by row to show the distribution of predictions for each true age group
    row_sums = heatmap.sum(axis=1, keepdims=True)
    heatmap_norm = np.divide(heatmap, row_sums, out=np.zeros_like(heatmap), where=row_sums != 0)

    # Create labels for the heatmap axes
    x_labels = [f'{int(xedges[i])}-{int(xedges[i + 1])}' if i < len(xedges) - 2 else f'{int(xedges[i])}+' for i in
                range(len(xedges) - 1)]
    y_labels = [f'{int(yedges[i])}-{int(yedges[i + 1])}' if i < len(yedges) - 2 else f'{int(yedges[i])}+' for i in
                range(len(yedges) - 1)]

    sns.heatmap(heatmap_norm, cmap='YlGnBu', annot=True, fmt=".2f",
                xticklabels=x_labels,
                yticklabels=y_labels)
    plt.xlabel('Predicted Age (years)')
    plt.ylabel('True Age (years)')
    plt.title('Distribution of Predictions by Age Group (Row Normalized)')

    plt.tight_layout()
    plt.savefig('age_prediction_analysis.png', dpi=300)
    plt.show()