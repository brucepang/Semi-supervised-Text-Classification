# Semi-supervised-Text-Classification

## Usage
For supervised learing, 
run sentiment.py with the following optional arguments,
--token: which tokenization method to use. Default is "count", short for countVectorizer. Another choice is "tf", short for tf-idf. 
--select_features: whether select k best features. Default is False.
--k: K value for select_features. Default is 5000.
--max_df: Used for tf-idf tokenization. Terms that have a document frequency strictly higher than that will be ignored when building the vocabulary. Default is 0.4.

For semi-supervise learning,
run semi-supervised.py with the following optional arguments,
--confidant: whether expand only confidant prediction. Default is False. 