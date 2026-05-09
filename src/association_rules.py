import os
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


def generate_mock_listening_data(num_samples=1000):
    # simulate a dataset of user listening sessions based on music psychology
    # attributes: [emotion, tempo, musical key, energy level]
    dataset = []
    
    for _ in range(num_samples):

        # generate random number between 0 and 1
        # used to simulate probability distributions
        rand = np.random.random()
        
        # generate transactions based on assumed psychological distributions
        if rand < 0.2:
            dataset.append(['happy', 'fast_tempo', 'major_key', 'high_energy'])
        
        elif rand < 0.4:
            dataset.append(['sad', 'slow_tempo', 'minor_key', 'low_energy'])
        
        elif rand < 0.55:
            dataset.append(['angry', 'fast_tempo', 'minor_key', 'high_energy'])
        
        elif rand < 0.7:
            dataset.append(['neutral', 'slow_tempo', 'major_key', 'low_energy'])
        
        elif rand < 0.8:
            dataset.append(['surprise', 'fast_tempo', 'major_key', 'high_energy'])
        
        elif rand < 0.9:
            dataset.append(['fear', 'fast_tempo', 'minor_key', 'low_energy'])
        
        else:
            dataset.append(['disgust', 'slow_tempo', 'minor_key', 'low_energy'])
            
        # add some random noise (anomalies) to make data realistic
        if np.random.random() < 0.1:
            dataset[-1][1] = 'slow_tempo' if dataset[-1][1] == 'fast_tempo' else 'fast_tempo'
            
    return dataset



def run_apriori_algorithm(dataset, results_dir):
    # one-hot encode the transaction dataset
    
    # apriori algorithm requires binary matrix format:
    # each column represents an item
    # each row represents a transaction

    # example:
    # happy | fast_tempo | major_key | high_energy
    #   1   |      1      |     1     |      1
    te = TransactionEncoder()
    
    # fit encoder and transform transactions
    te_ary = te.fit(dataset).transform(dataset)

    # convert encoded matrix into dataframe
    df = pd.DataFrame(te_ary, columns=te.columns_)
    
    # apply apriori to find frequent itemsets (min support 5%)
    frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True)
    
    # generate association rules from frequent itemsets (min confidence 70%)

    # example:
    # happy -> fast_tempo
    # confidence = 0.85 means:
    # 85% of happy sessions also contain fast_tempo
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
    
    # extract pure strings from frozensets for easier filtering and reading
    rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x)[0] if len(x) == 1 else list(x))
    rules['consequents'] = rules['consequents'].apply(lambda x: list(x)[0] if len(x) == 1 else list(x))
    
    emotions = ['happy', 'sad', 'angry', 'neutral', 'surprise', 'fear', 'disgust']
    
    # filter rules to show only Emotion -> Music Attribute relationships
    # ensure antecedents is a string (single emotion) before filtering
    emotion_rules = rules[
        rules['antecedents'].apply(lambda x: isinstance(x, str) and x in emotions)
    ]
    
    # sort by confidence (highest probability rules first)
    emotion_rules = emotion_rules.sort_values(by='confidence', ascending=False)
    
    # select columns for the final report
    # support: frequency of rule in entire dataset
    # confidence: probability of consequent given antecedent
    # lift: strength of relationship compared to random chance
    final_table = emotion_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    
    csv_path = os.path.join(results_dir, 'music_association_rules.csv')
    final_table.to_csv(csv_path, index=False)
    
    return final_table

if __name__ == "__main__":
    print("Starting association rules generation (Apriori algorithm)...")
    
    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. generate statistical data
    print("Generating synthetic listening sessions dataset...")
    transactions = generate_mock_listening_data(num_samples=1500)
    
    # 2. run apriori
    print("Mining association rules...")
    rules_table = run_apriori_algorithm(transactions, results_dir)
    
    # 3. display the mapping table
    print("\nEmotion to music attributes mapping (confidence > 70%):")
    print(rules_table.to_string(index=False))
    
    print(f"\nProcess completed. Rules saved to {results_dir}/music_association_rules.csv")