import pandas as pd
import numpy as np
import ast
from sklearn.metrics import precision_recall_fscore_support

def get_result(path):
    processed_df = pd.read_csv(path)
    processed_df['Tag'] = processed_df['Tag'].apply(ast.literal_eval)
    processed_df['Predicted Tag'] = processed_df['Predicted Tag'].apply(ast.literal_eval)

    actual_tags = processed_df['Tag']
    predicted_tags = processed_df['Predicted Tag']

    # Flatten lists
    actual_tags = [tag for tag_list in actual_tags for tag in tag_list]
    predicted_tags = [tag for tag_list in predicted_tags for tag in tag_list]

    # Calculate precision, recall, and F1-score for each tag
    precision, recall, f1, support = precision_recall_fscore_support(actual_tags, predicted_tags, average=None, labels=np.unique(actual_tags))

    # Map tags to support counts
    count = dict()
    for i, tag in enumerate(np.unique(actual_tags)):
        count[tag] = support[i]

    # Map these metrics to each unique tag
    tag_metrics = dict()
    unique_tags = np.unique(actual_tags)
    for i, tag in enumerate(unique_tags):
        tag_metrics[tag] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1_score': f1[i],
            "support": count[tag]
        }
    stats = pd.DataFrame(tag_metrics).T
    stats = stats[:-1] #remove the O tag row
    print(stats)
    #get weighted average of each column by support
    weighted_avg = stats.apply(lambda x: np.average(x, weights=stats["support"]))
    print(weighted_avg)
    # Display the metrics for each tag
    for tag, metrics in tag_metrics.items():
        print(f"Tag: {tag}  Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1 Score: {metrics['f1_score']:.4f}, Support: {metrics['support']}")

    # Construct labels unique tags absent the O tag (need to actually find the O tag in the list first)
    unique_tags = np.unique(actual_tags)
    unique_tags = unique_tags[unique_tags != 'O']

    # Now compute overall precision, recall, and F1-score excluding the O tag
    precision, recall, f1, support = precision_recall_fscore_support(actual_tags, predicted_tags, average='weighted', labels=unique_tags)
    print(f"Overall Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

# Get results for each model: early_stopping, e50, e100
get_result("results/early_stopping_df.csv")
print('----------------------------------------')
get_result("results/e50_df.csv")
print('----------------------------------------')
get_result("results/e100_df.csv")