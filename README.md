# tfdsbd

Deep Sentence Boundary Detector implemented with TensorfFlow using.


## Usage
```python
from tfdsbd import SentenceBoundaryDetector

# By default uses multylanguage pretrained model.
sbd = SentenceBoundaryDetector()
# But you can supply 'en', 'ru' flags or path to your custom exported model

raw_documents = ['...', '...']
splitted_dicuments = sbd.split(raw_documents)
print(splitted_dicuments)
```

## Model architecture

1. Split raw input text into tokens [with unicode rules](http://unicode.org/reports/tr29/#Word_Boundaries). This leaves space-like characters as separate tokens.
2. Wrap tokens with '<' and '>' as begin/end markers and split them into ngrams. We leave in main vocabulary only begin, end and not alpha-num ngrams.
3. Embed and combine ngrams preserving size of embeddings.
4. Apply RNN
5. Compute logits
6. Compute sigmoid cross-enropy.

## Default hyperparameters

Most important hyperparameters used for pretrained models:
- ngrams: from 3 to 5
- embeddings: mean of ngram embeddings with size 20
- rnn: 1-layer bidirectional LSTM with 32 nodes

## Training custom model

1. Obtain dataset with already splitted sentences.

If order matters sentences should be separated by single "\n". This calls "paragraph" and it should not be very long.

Paragraphs (including single sentences) should be separated with at least double "\n".


Here is a simple dataset example:
```
Single sentence.

First sentence in paragraph.
Second sentence in paragraph.
And finally third one!

One more single sentence in paragraph.

```

2. Prepare configuration file with hyperparameters. Start from config/default.json in this module repository.

3. Split dataset into train and eval.
```bash
tfdsbd-split data/dataset.txt data/train.txt dataset/eval.txt
```

4. Convert datasets to TFRecords.
```bash
tfdsbd-dataset data/train.txt train/
tfdsbd-dataset data/eval.txt eval/
```

5. Extract most frequent non-aplphanum ngrams vocabulary from train dataset.
```bash
tfdsbd-vocab train/ config/default.json train/vocabulary.pkl
```

6. Run training. First run will only compute baseline metrics, so you should run repeat this stem multiple times.
```bash
tfdsbd-train train/ train/vocabulary.pkl config/default.json model/ -eval_data eval/ -export_path export/
```

7. Test your model on plain text file.
```bash
tfdsbd-infer export/<model_version> some_text_document.txt
```


## No training
{'accuracy': 0.7127404, 'accuracy_baseline': 0.96334475, 'auc': 0.43357098, 'auc_precision_recall': 0.029233117, 'average_loss': 0.6477022, 'label/mean': 0.03665527, 'loss': 14295.649, 'precision': 0.020550122, 'prediction/mean': 0.47137526, 'recall': 0.14651878, 'global_step': 1, 'f1': 0.036044756812566914}

## Bidirectional 1-layer GRU of size 1
{'accuracy': 0.96334475, 'accuracy_baseline': 0.96334475, 'auc': 0.5723163, 'auc_precision_recall': 0.09823781, 'average_loss': 0.31047523, 'label/mean': 0.03665527, 'loss': 3426.3008, 'precision': 0.0, 'prediction/mean': 0.23483902, 'recall': 0.0, 'global_step': 25234}


