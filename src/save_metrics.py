import os
import csv
from datetime import datetime


METRICS_CSV = "../results/model_metrics.csv"

# column headers for the master metrics table
FIELDNAMES = [
    'model_name',
    'dataset',
    'accuracy',
    'macro_f1',
    'macro_precision',
    'macro_recall',
    'notes',
    'timestamp'
]


def save_metrics(model_name, dataset, accuracy, macro_f1,
                 macro_precision=None, macro_recall=None, notes=""):
    # append one row of results to the shared model_metrics.csv file
    # this allows all scripts to write to a single comparison table

    os.makedirs(os.path.dirname(METRICS_CSV), exist_ok=True)

    # check whether file exists to decide if header row is needed
    file_exists = os.path.exists(METRICS_CSV)

    with open(METRICS_CSV, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)

        if not file_exists:
            # write header only once when file is first created
            writer.writeheader()

        writer.writerow({
            'model_name': model_name,
            'dataset': dataset,
            'accuracy': round(accuracy, 4),
            'macro_f1': round(macro_f1, 4),
            'macro_precision': round(macro_precision, 4) if macro_precision is not None else '',
            'macro_recall': round(macro_recall, 4) if macro_recall is not None else '',
            'notes': notes,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
        })

    print(f"  [metrics saved] {model_name}: acc={accuracy:.4f}, f1={macro_f1:.4f}")
