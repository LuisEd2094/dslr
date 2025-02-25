import matplotlib.pyplot as plt
import sys
import math


def get_csv(filename="dataset_train.csv"):
    with open(filename, "r") as file:
        for line in file:
            yield line

def read_csv(file):
    data = []
    for line in file:
        data.append(line.strip().split(","))
    return data

def get_args():
    if len(sys.argv) != 2:
        raise Exception("Usage: python describe.py <filename>")
    return sys.argv[1]

def clean_features(features):

    to_remove = []
    for column in features:
        feature = features[column]
        if len(feature) == 0 or all(math.isnan(x) for x in feature):
            to_remove.append(column)
    return {k: v for k, v in features.items() if k not in to_remove}

def get_features(filename="dataset_train.csv"):
    csv = get_csv(filename)
    csv = read_csv(csv)

    # Skip the first column (assuming it's an index)
    features = {header: [] for header in csv[0][1:]}
    for row in csv[1:]:  # Skip the header row
        for i, value in enumerate(row[1:], start=1):
            column = csv[0][i]
            try:
                value = float(value)
                features[column].append(value)
            except ValueError:
                pass

    return clean_features(features)

def get_histogram(features):
    num_bins = 10  # Number of bins per histogram
    num_features = len(features)
    cols = 3
    rows = math.ceil(num_features / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()
    for i, (feature_name, data) in enumerate(features.items()):
        min_val, max_val = min(data), max(data)
        bin_width = (max_val - min_val) / num_bins
        bins = [min_val + j * bin_width for j in range(num_bins + 1)]

        # Count occurrences in each bin
        bin_counts = [0] * num_bins
        for value in data:
            for k in range(num_bins):
                if bins[k] <= value < bins[k + 1]:
                    bin_counts[k] += 1
                    break

        # Plot histogram
        axes[i].bar(
            bins[:-1], bin_counts, width=bin_width, edgecolor="black", alpha=0.7
        )
        axes[i].set_title(feature_name)
        axes[i].set_xlabel("Value Range")
        axes[i].set_ylabel("Frequency")

    # Hide any unused subplots if there are extra spaces
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig("histograms.png", dpi=300)

if __name__ == "__main__":
    get_histogram(get_features())
