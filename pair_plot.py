import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) != 2:
        print("Usage: python pair_plot.py dataset_train.csv")
        sys.exit(1)
    
    file_path = sys.argv[1]
    df = pd.read_csv(file_path)

    # Extract numerical features and the Hogwarts House column
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    df_plot = df[numeric_cols + ['Hogwarts House']]

    # Generate pair plot
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="ticks")
    plot = sns.pairplot(
        df_plot,
        hue='Hogwarts House',
        palette='viridis',
        plot_kws={'alpha': 0.6},
        diag_kind='hist'
    )
    plot.figure.suptitle("Pair Plot of Numerical Features by Hogwarts House", y=1.02)
    
    # plt.show()
    plt.savefig("pair_plot.png", bbox_inches='tight')
    print("Pair plot saved to pair_plot.png")

if __name__ == "__main__":
    main()