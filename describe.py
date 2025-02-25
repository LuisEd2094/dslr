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


def clean_data(csv, descriptions):
    features = {header: {desc: [] for desc in descriptions} for header in csv[0]}
    for row in csv[1:]:
        for i, value in enumerate(row):
            column = csv[0][i]
            try:
                value = float(value)
                features[column]["summary"].append(value)
            except ValueError:
                pass
    to_remove = []
    for column in features:
        feature = features[column]["summary"]
        if len(feature) == 0 or all(math.isnan(x) for x in feature):
            to_remove.append(column)
    return {k: v for k, v in features.items() if k not in to_remove}


def get_args():
    if len(sys.argv) != 2:
        raise Exception("Usage: python describe.py <filename>")
    return sys.argv[1]


def percentile(data, percent):
    index = (percent / 100) * (len(data) - 1)
    lower = int(index)
    upper = lower + 1

    if upper >= len(data):
        return data[lower]  # If exact index, return value
    weight = index - lower

    return data[lower] + weight * (data[upper] - data[lower])


# Using same formula as Pandas for kurtosis
# Kurtosis measures how extreme the outliers are (how heavy or light the tails are).
# Kurtosis ≈ 0 → Mesokurtic (normal tails, like a normal distribution)
# Kurtosis > 0 → Leptokurtic (heavy tails, more outliers than normal)
# Kurtosis < 0 → Platykurtic (light tails, fewer extreme values than normal)
def kurtosis(mean, std, len, data):
    numerator = sum(((x - mean) / std) ** 4 for x in data)
    excess_kurtosis = (
        len * (len + 1) / ((len - 1) * (len - 2) * (len - 3))
    ) * numerator - (3 * (len - 1) ** 2 / ((len - 2) * (len - 3)))
    return excess_kurtosis


# Using same formula as Pandas for skewness
# Skewness measures whether data is balanced around the mean or shifted to one side.
# If skewness = 0 → Data is perfectly symmetric (normal distribution).
# If skewness > 0 → Data is right-skewed (positive skew) (longer tail on the right, we have some big values to the right).
# If skewness < 0 → Data is left-skewed (negative skew) (longer tail on the left, we have some small values to the left).
def skewness(mean, std, len, data):
    return (len / ((len - 1) * (len - 2))) * sum(((x - mean) / std) ** 3 for x in data)


def get_features(filename="dataset_train.csv"):
    descriptions = [
        "summary",
        "count",
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
        "skew",
        "kurtosis",
        "var",
        "range",
    ]
    csv = get_csv(filename)
    csv = read_csv(csv)
    features = clean_data(csv, descriptions)
    for column in features:
        summary = features[column]["summary"]
        features[column]["count"] = len(summary)
        features[column]["mean"] = sum(summary) / features[column]["count"]

        # Variance and standard deviation are very similar.
        # Variance is the average of the squared differences from the Mean.
        # Standard deviation is the square root of the variance.
        # STD is more commonly used because it is in the same units as the data.
        # Variance is in squared units, which is not as intuitive, but some formulas require it, since its easier to work with.
        features[column]["var"] = sum(
            [(x - features[column]["mean"]) ** 2 for x in summary]
        ) / (features[column]["count"] - 1)

        features[column]["std"] = features[column]["var"] ** 0.5
        sorted_summary = sorted(summary)

        features[column]["min"] = sorted_summary[0]
        features[column]["max"] = sorted_summary[-1]
        features[column]["25%"] = percentile(sorted_summary, 25)
        features[column]["50%"] = percentile(sorted_summary, 50)
        features[column]["75%"] = percentile(sorted_summary, 75)

        features[column]["skew"] = skewness(
            features[column]["mean"],
            features[column]["std"],
            features[column]["count"],
            summary,
        )
        features[column]["kurtosis"] = kurtosis(
            features[column]["mean"],
            features[column]["std"],
            features[column]["count"],
            summary,
        )
        features[column]["range"] = features[column]["max"] - features[column]["min"]

    return features


if __name__ == "__main__":
    try:
        args = get_args()
        df = get_features(args)
        for column in df:
            summary = df[column]["summary"]
            if len(summary) == 0 or all(math.isnan(x) for x in summary):
                continue
            print(f"{column}:")
            for desc in df[column]:
                if desc != "summary":
                    print(f"  {desc}: {df[column][desc]:.6f}")
    except Exception as e:
        print(e)
        sys.exit(1)
