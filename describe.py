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

def percentile(data, percent):
    index = (percent / 100) * (len(data) - 1)
    lower = int(index)
    upper = lower + 1

    if upper >= len(data):
        return data[lower]  # If exact index, return value
    weight = index - lower
    
    return data[lower] + weight * (data[upper] - data[lower])


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
    ]
    csv = get_csv(filename)
    csv = read_csv(csv)
    features = {header: {desc: [] for desc in descriptions} for header in csv[0]}
    for row in csv[1:]:
        for i, value in enumerate(row):
            column = csv[0][i]
            try:
                value = float(value)
                features[column]["summary"].append(value)
            except ValueError:
                pass
    for column in features:
        summary = features[column]["summary"]
        if len(summary) == 0 or all(math.isnan(x) for x in summary):
            continue
        features[column]["count"] = len(summary)
        features[column]["mean"] = sum(summary) / features[column]["count"]
        features[column]["std"] = (
            sum([(x - features[column]["mean"]) ** 2 for x in summary])
            / (features[column]["count"] - 1)
        ) ** 0.5

        sorted_summary = sorted(summary)
        features[column]["min"] = sorted_summary[0]
        features[column]["max"] = sorted_summary[-1]
        features[column]["25%"] = percentile(sorted_summary, 25)
        features[column]["50%"] = percentile(sorted_summary, 50)
        features[column]["75%"] = percentile(sorted_summary, 75)

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
