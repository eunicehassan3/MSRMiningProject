

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import mean
import csv
from csv import DictReader

def main():
    # Load the CSV file into a pandas DataFrame
    average_population_metric("mining_project_data.csv", "updated_mining_project_data.csv")
    # print(new_file)
    df = pd.read_csv('updated_mining_project_data.csv')
    print(df)


    # Calculate the correlation matrix
    correlation_matrix = df[['total_releases' , 'aggregate_score', 'speed_metric', 'average_popularity_metric']].corr()
    # correlation_matrix = df[['popularity_score', 'total_popularity_1_year', 'average_freshness', 'average_speed', 'dependencies_count']].corr()
    #
    # Plot the correlation matrix using a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Popularity Metrics and Dependencies Count')
    plt.show()


def average_population_metric(file, output_csv):
    with open(file, newline='') as f:
        reader = DictReader(f)
        rows = list(reader)

        for row in rows:
            popularity_str = row['popularity_metrics']
            popularity_list = eval(row['popularity_metrics'])
            average_popularity = sum(popularity_list) / len(popularity_list) if popularity_list else 0.0
            row['average_popularity_metric'] = (average_popularity)

    with open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
        fieldnames = reader.fieldnames + ['average_popularity_metric']  # Add the new column to the fieldnames
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        writer.writeheader()  # Write the header row (including the new column)
        writer.writerows(rows)  # Write the updated rows




main()