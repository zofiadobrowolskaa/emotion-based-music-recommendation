import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


def generate_mock_listening_data(num_samples=1500):
    # simulate a dataset of user listening sessions based on music psychology research
    # attributes per session: [emotion, tempo, musical key, energy level]
    #
    # note: real user listening data is not available, so we generate synthetic sessions
    # using well-established music psychology mappings (see bibliography: Grekow, 2016)
    # each session is a transaction containing: emotion + music attributes co-listened
    dataset = []

    for _ in range(num_samples):

        # generate random number between 0 and 1
        # used to simulate realistic probability distributions across emotions
        rand = np.random.random()

        # assign music attributes according to emotion-specific audio profiles
        if rand < 0.2:
            # happy music tends to be fast, major, and energetic (valence+, arousal+)
            dataset.append(['happy', 'fast_tempo', 'major_key', 'high_energy'])

        elif rand < 0.4:
            # sad music tends to be slow, minor, and low-energy (valence-, arousal-)
            dataset.append(['sad', 'slow_tempo', 'minor_key', 'low_energy'])

        elif rand < 0.55:
            # angry music tends to be fast, minor, and high-energy (valence-, arousal+)
            dataset.append(['angry', 'fast_tempo', 'minor_key', 'high_energy'])

        elif rand < 0.7:
            # neutral music tends to be calm and balanced
            dataset.append(['neutral', 'slow_tempo', 'major_key', 'low_energy'])

        elif rand < 0.8:
            # surprise music matches energetic and upbeat patterns
            dataset.append(['surprise', 'fast_tempo', 'major_key', 'high_energy'])

        elif rand < 0.9:
            # fear music tends to be tense: fast, minor, but lower energy
            dataset.append(['fear', 'fast_tempo', 'minor_key', 'low_energy'])

        else:
            # disgust is mapped to slow, dark, low energy
            dataset.append(['disgust', 'slow_tempo', 'minor_key', 'low_energy'])

        # add controlled noise (10% chance) to simulate real-world variability
        # (e.g., user listens to surprising tempo for their emotion)
        if np.random.random() < 0.1:
            dataset[-1][1] = 'slow_tempo' if dataset[-1][1] == 'fast_tempo' else 'fast_tempo'

    return dataset


def run_apriori_algorithm(dataset, results_dir):
    # one-hot encode the transaction dataset

    # apriori algorithm requires binary matrix format:
    # each column represents an item, each row represents a transaction
    # example:
    # happy | fast_tempo | major_key | high_energy
    #   1   |      1     |     1     |      1
    te = TransactionEncoder()

    # fit encoder and transform all transactions into binary matrix
    te_ary = te.fit(dataset).transform(dataset)

    # convert encoded matrix into pandas dataframe
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # apply apriori to find frequent itemsets appearing in at least 5% of transactions
    frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True)

    # generate association rules with minimum 70% confidence
    # confidence = P(consequent | antecedent)
    # example: confidence(happy -> fast_tempo) = 0.85
    # means 85% of "happy" sessions also contain "fast_tempo"
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

    # convert frozensets to plain strings or lists for readability
    rules['antecedents'] = rules['antecedents'].apply(
        lambda x: list(x)[0] if len(x) == 1 else list(x)
    )
    rules['consequents'] = rules['consequents'].apply(
        lambda x: list(x)[0] if len(x) == 1 else list(x)
    )

    emotions = ['happy', 'sad', 'angry', 'neutral', 'surprise', 'fear', 'disgust']

    # filter to show only emotion -> music attribute rules
    # (we don't want rules like fast_tempo -> high_energy)
    emotion_rules = rules[
        rules['antecedents'].apply(lambda x: isinstance(x, str) and x in emotions)
    ]

    # sort by confidence so strongest rules appear first
    emotion_rules = emotion_rules.sort_values(by='confidence', ascending=False)

    # select relevant columns for the final report table
    final_table = emotion_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

    csv_path = os.path.join(results_dir, 'music_association_rules.csv')
    final_table.to_csv(csv_path, index=False)

    return final_table


def plot_rules_heatmap(rules_table, results_dir):
    # create a confidence heatmap: emotion (rows) vs music attribute (columns)
    # this gives an intuitive overview of which attributes are associated with each emotion

    emotions = ['happy', 'sad', 'angry', 'neutral', 'surprise', 'fear', 'disgust']

    music_attributes = ['fast_tempo', 'slow_tempo', 'major_key', 'minor_key', 'high_energy', 'low_energy']

    # build a pivot matrix filled with 0 by default
    heatmap_data = pd.DataFrame(0.0, index=emotions, columns=music_attributes)

    for _, row in rules_table.iterrows():
        emotion = row['antecedents']
        consequent = row['consequents']

        if isinstance(emotion, str) and emotion in emotions:
            # handle both single string and list consequents
            attrs = [consequent] if isinstance(consequent, str) else consequent
            for attr in attrs:
                if attr in music_attributes:
                    # fill cell with rule confidence value
                    heatmap_data.loc[emotion, attr] = round(row['confidence'], 2)

    plt.figure(figsize=(12, 6))

    sns.heatmap(
        heatmap_data,
        annot=True,          # show confidence values in cells
        fmt='.2f',           # two decimal places
        cmap='YlOrRd',       # yellow-to-red gradient (higher = stronger rule)
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor='gray'
    )

    plt.title('Association Rules: Emotion → Music Attribute Confidence', fontsize=14)
    plt.xlabel('Music Attribute', fontsize=11)
    plt.ylabel('Detected Emotion', fontsize=11)
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()

    plt.savefig(os.path.join(results_dir, 'association_rules_heatmap.png'), dpi=150)
    plt.close()

    print("Association rules heatmap saved.")


if __name__ == "__main__":
    print("Starting association rules generation (Apriori algorithm)...")

    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)

    # 1. generate synthetic listening session data
    print("Generating synthetic listening sessions dataset (n=1500)...")
    transactions = generate_mock_listening_data(num_samples=1500)

    # 2. run apriori and extract emotion -> music attribute rules
    print("Mining association rules with Apriori algorithm...")
    rules_table = run_apriori_algorithm(transactions, results_dir)

    # 3. visualize rules as confidence heatmap
    plot_rules_heatmap(rules_table, results_dir)

    # 4. display the mapping table
    print("\nEmotion to music attributes mapping (confidence > 70%):")
    print(rules_table.to_string(index=False))

    print(f"\nProcess completed. Rules saved to {results_dir}/music_association_rules.csv")
