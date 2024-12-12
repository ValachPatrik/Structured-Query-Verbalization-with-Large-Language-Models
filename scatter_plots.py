import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
'''
Given a results csv, this file creates a scatter plot between bert score of query and NL embedding (or pairwise
 distance) vs.bert score of NL and NL_gt. The higher the correlation, the better the filter works.
'''

# Load the CSV file
file_path = 'data/v2/translation_results.csv'
df = pd.read_csv(file_path)

# Scatter plot function with regression line and correlation coefficient
def scatter_plot_with_regression(df, x_col, y_col, title):
    # Extract x and y data
    x = df[x_col]
    y = df[y_col]

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    line = slope * x + intercept

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Data points', color='blue', alpha=0.6)
    plt.plot(x, line, color='red', label=f'Regression line (r={r_value:.2f})')

    # Plot settings
    plt.xlabel(x_col, fontsize=14)
    plt.ylabel(y_col, fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()

# Scatter plot: bert_q_NL vs bert_NL_NL_gt
scatter_plot_with_regression(df, 'bert_q_NL', 'bert_NL_NL_gt', 'Scatter Plot: bert_q_NL vs bert_NL_NL_gt')

# Scatter plot: intra_cluster_distance vs bert_NL_NL_gt
scatter_plot_with_regression(df, 'intra_cluster_distance', 'bert_NL_NL_gt', 'Scatter Plot: intra_cluster_distance vs bert_NL_NL_gt')
