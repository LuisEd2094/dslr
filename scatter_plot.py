import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    if len(sys.argv) != 2:
        print("Usage: python scatter_plot.py dataset_train.csv")
        sys.exit(1)
    
    file_path = sys.argv[1]
    df = pd.read_csv(file_path)

    # Extract numerical features (exclude non-numeric columns)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) < 2:
        print("Error: Not enough numerical features in the dataset.")
        sys.exit(1)

    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr().abs()

    # Find the pair with the highest correlation (excluding self-correlation)
    max_corr = 0
    feature1, feature2 = None, None
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            current_corr = corr_matrix.iloc[i, j]
            if current_corr > max_corr:
                max_corr = current_corr
                feature1 = corr_matrix.columns[i]
                feature2 = corr_matrix.columns[j]

    # Plot the scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x=feature1,
        y=feature2,
        hue='Hogwarts House',
        palette='viridis',
        alpha=0.8
    )
    plt.title(f"Most Similar Features: {feature1} vs {feature2} (Correlation: {max_corr:.2f})")
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.legend(title='Houses', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    # plt.show()
    plt.savefig("scatter_plot.png")
    print("Plot saved to scatter_plot.png")

if __name__ == "__main__":
    main()