import pandas as pd
import json
import colorama

# Read CSV file
df = pd.read_csv('./train.csv')

print("数据集条数:", df.shape[0])


# Convert JSON formatted 'udmap' to dictionary, handling 'unknown' values
def convert_udmap(udmap_str):
    try:
        return json.loads(udmap_str)
    except json.JSONDecodeError:
        return {'unknown': True}


df['udmap'] = df['udmap'].apply(convert_udmap)

# Use explode() to expand the udmap column and value_counts() to get feature counts
udmap_features = df['udmap'].explode().value_counts()

# If there are 'unknown' values, subtract their count from the feature count
if 'unknown' in udmap_features:
    unknown_count = udmap_features['unknown']
    udmap_features = udmap_features.drop('unknown')
else:
    unknown_count = 0

# Calculate the total number of samples in the dataset (excluding 'unknown' samples)
total_samples_excluding_unknown = df.shape[0] - unknown_count

# Calculate the percentage for each key
key_percentages = (udmap_features / total_samples_excluding_unknown) * 100

# Calculate the percentage for 'unknown' values
unknown_percentage = (unknown_count / df.shape[0]) * 100

# Output the feature counts, unknown count, and key percentages
print("Feature counts:")
print(udmap_features)
print("Unknown count:", unknown_count)

print("\nKey percentages:")
print(key_percentages)

print("Unknown percentage:", unknown_percentage)


def show_features_is_up_together(key1: int, key2: int):
    # Initialize count for key3 and key2 appearing together
    key1 = "key" + str(key1)
    key2 = "key" + str(key2)
    key1_and_key2_count = 0

    # Loop through each record in the DataFrame
    for udmap_dict in df['udmap']:
        # Check if both 'key3' and 'key2' are present in the record
        if key1 in udmap_dict and key2 in udmap_dict:
            key1_and_key2_count += 1

    # Calculate the probability of 'key3' and 'key2' appearing together
    probability_key3_and_key2 = key1_and_key2_count / total_samples_excluding_unknown

    # Output the result
    print(f"{key1} 和 {key2} 同时出现的概率：", probability_key3_and_key2)
    return key1_and_key2_count


def show_features_is_not_up_together(key1: int, key2: int):
    key1 = "key" + str(key1)
    key2 = "key" + str(key2)
    key1_or_key2_only_count = 0

    # Loop through each record in the DataFrame
    for udmap_dict in df['udmap']:
        # Check if 'key2' is present and 'key3' is missing, or vice versa
        if (key1 in udmap_dict and key2 not in udmap_dict) or (key1 not in udmap_dict and key2 in udmap_dict):
            key1_or_key2_only_count += 1

    # Calculate the probability of only 'key2' or only 'key3' appearing
    probability_key2_or_key3_only = key1_or_key2_only_count / total_samples_excluding_unknown

    # Output the result
    print(f"仅出现 {key1} 或 {key2} 的概率：", probability_key2_or_key3_only)


def get_if_features_up(key1: int, key2: int):
    key1 = "key" + str(key1)
    key2 = "key" + str(key2)
    key1_or_key2_count = 0

    # Loop through each record in the DataFrame
    for udmap_dict in df['udmap']:
        # Check if 'key2' is present and 'key3' is missing, or vice versa
        if key1 in udmap_dict or key2 in udmap_dict:
            key1_or_key2_count += 1

    # Calculate the probability of only 'key2' or only 'key3' appearing
    probability_key2_or_key3_only = key1_or_key2_count / total_samples_excluding_unknown

    # Output the result
    print(f"只要出现 {key1} 或 {key2} 的概率：", probability_key2_or_key3_only)
    return key1_or_key2_count


def key5_():
    key1_or_key2_count = 0

    key1 = "key1"
    key2 = "key2"
    key3 = "key3"
    key4 = "key4"
    key5 = "key5"

    # Loop through each record in the DataFrame
    for udmap_dict in df['udmap']:
        # Check if 'key2' is present and 'key3' is missing, or vice versa
        if key1 in udmap_dict and key2 in udmap_dict and key3 in udmap_dict and key4 in udmap_dict and key5 in udmap_dict:
            key1_or_key2_count += 1

    # Calculate the probability of only 'key2' or only 'key3' appearing
    probability_key2_or_key3_only = key1_or_key2_count / total_samples_excluding_unknown

    # Output the result
    print("出现key12345的数量:", key1_or_key2_count)


if __name__ == '__main__':
    base = get_if_features_up(4, 5)
    uo = show_features_is_up_together(4, 5)
    print(colorama.Fore.LIGHTGREEN_EX)
    print("key4 and key5 show up together in if key2 or key3 show up in dataset:", (uo / base) * 100, "%")
    key5_()
