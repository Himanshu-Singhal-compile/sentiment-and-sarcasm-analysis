# Sentiment and Sarcasm Analysis

Two NLP notebook studies: binary Twitter sentiment classification with Word2Vec and LSTM, and sarcasm detection comparing learned Word2Vec embeddings with pretrained GloVe.

[![CI](https://github.com/Himanshu-Singhal-compile/sentiment-and-sarcasm-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/Himanshu-Singhal-compile/sentiment-and-sarcasm-analysis/actions/workflows/ci.yml)


## Environment

Use Python 3.11 in an isolated environment. The version ranges in `requirements.txt`
describe the intended environment; they are not a fully resolved lockfile.

```sh
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyter lab
```

Launch Jupyter from the repository root and select a notebook in `notebooks/`.
Data defaults to `data/`; set `PROJECT_DATA_DIR` to override it. Generated artifacts
are ignored by Git. Clear notebook outputs before committing.

## Notebooks

| Notebook | Workflow |
| --- | --- |
| `notebooks/twitter_sentiment.ipynb` | Text cleaning → Word2Vec → tokenization → frozen embeddings → LSTM → sentiment score |
| `notebooks/sarcasm_detection.ipynb` | Headline cleaning → Word2Vec/BiLSTM/GRU and GloVe/BiLSTM → binary sarcasm evaluation |

Compatibility updates use TensorFlow Keras imports, Gensim 4 vocabulary/vector APIs, current optimizer argument names, thresholded predictions, and explicit local data paths. The environment targets TensorFlow 2.15 on Python 3.11; Python 3.13 is not the documented training environment.

## Data and resources

Download these separately, subject to their respective terms:

| File in `data/` | Source |
| --- | --- |
| `training.1600000.processed.noemoticon.csv` | [Sentiment140](http://help.sentiment140.com/for-students) |
| `Sarcasm_Headlines_Dataset_v2.json` | [News Headlines for Sarcasm Detection](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection) |
| `glove.twitter.27B.200d.txt` | [Stanford GloVe](https://nlp.stanford.edu/projects/glove/) |

Download the NLTK stopwords resource before running:

```sh
python -m nltk.downloader stopwords
python scripts/validate_notebooks.py
```

Full Sentiment140 contains 1.6 million tweets. Word2Vec training, sequence arrays, recurrent models, and the large GloVe lookup can require significant memory and runtime. No pretrained model, dataset, or credentials are included. The sentiment notebook saves trained assets to `artifacts/`.

## Evaluation and interpretation

Sentiment training expects labels 0 and 4 only. A neutral prediction is a confidence-band heuristic, not a separately trained neutral class. Recurrent-network outputs are scores, not calibrated probabilities.

The sarcasm Word2Vec branch builds its vocabulary/embeddings before the supervised split, and both branches use the test partition for validation during fitting. Treat those notebook comparisons as exploratory. For a defensible benchmark, split train/validation/test before learning vocabulary or embeddings and reserve test data until the final evaluation. Existing narrative accuracy statements are historical notebook text and were not independently reproduced during packaging; outputs were cleared.

## Attribution and validation

The supplied notebooks contain tutorial-style material and an existing source credit in the sarcasm notebook. Those credits are retained. An original upstream notebook URL/author was not supplied, so this repository does not claim that every notebook passage or implementation is original. Confirm upstream attribution and reuse terms before relicensing.

CI checks notebook JSON and Python syntax. Full neural training was not run during packaging because the required datasets and embeddings were not included. This repository provides research notebooks, without an API or production model service.


## Contributing and provenance

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow and
[NOTICE.md](NOTICE.md) for source provenance and reuse terms. Dataset licenses
are separate from code rights. No open-source license was present in the supplied
materials, so no blanket license has been added.
