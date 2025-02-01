![Pashto Word Cloud](https://github.com/dr-ijaz/nlpashto/blob/main/wc.gif)

# NLPashto – NLP Toolkit for Pashto

![GitHub](https://img.shields.io/github/license/dr-ijaz/nlpashto)
![GitHub contributors](https://img.shields.io/github/contributors/dr-ijaz/nlpashto) 
![Code size](https://img.shields.io/github/languages/code-size/dr-ijaz/nlpashto)
[![PyPI](https://img.shields.io/pypi/v/nlpashto)](https://pypi.org/project/nlpashto/)
[![Citation](https://img.shields.io/badge/citation-10-green)](https://scholar.google.com.pk/scholar?cites=14346943839777454210)

NLPashto is a Python suite for Pashto Natural Language Processing. It provides tools for fundamental text processing tasks, such as text cleaning, tokenization, and chunking (word segmentation). Additionally, it includes state-of-the-art models for POS tagging and sentiment analysis (specifically offensive language detection).

## Prerequisites

To use NLPashto, you will need:

- Python 3.8+

## Installing NLPashto

Install NLPashto via PyPi:

```bash
pip install nlpashto
```

## Basic Usage

### Text Cleaning

This module contains basic text cleaning utilities:

```python
from nlpashto import Cleaner

cleaner = Cleaner()
noisy_txt = "په ژوند کی علم 📚🖖 , 🖊  او پيسي 💵.  💸💲 دواړه حاصل کړه پوهان به دی علم ته درناوی ولري اوناپوهان به دي پیسو ته... https://t.co/xIiEXFg"

cleaned_text = cleaner.clean(noisy_txt)
print(cleaned_text)
# Output: په ژوند کی علم , او پيسي دواړه حاصل کړه پوهان به دی علم ته درناوی ولري او ناپوهان به دي پیسو ته
```

Parameters of the `clean` method:

- `text` (str or list): Input noisy text to clean.
- `split_into_sentences` (bool): Split text into sentences.
- `remove_emojis` (bool): Remove emojis.
- `normalize_nums` (bool): Normalize Arabic numerals (1, 2, 3, ...) to Pashto numerals (۱، ۲، ۳، ...).
- `remove_puncs` (bool): Remove punctuations.
- `remove_special_chars` (bool): Remove special characters.
- `special_chars` (list): List of special characters to keep.

### Tokenization (Space Correction)

This module corrects space omission and insertion errors. It removes extra spaces and inserts necessary ones:

```python
from nlpashto import Tokenizer

tokenizer = Tokenizer()
noisy_txt = 'جلال اباد ښار کې هره ورځ لس ګونه کسانپهډلهییزهتوګهدنشهيي توکو کارولو ته ا د ا م ه و رک وي'

tokenized_text = tokenizer.tokenize(noisy_txt)
print(tokenized_text)
# Output: [['جلال', 'اباد', 'ښار', 'کې', 'هره', 'ورځ', 'لسګونه', 'کسان', 'په', 'ډله', 'ییزه', 'توګه', 'د', 'نشه', 'يي', 'توکو', 'کارولو', 'ته', 'ادامه', 'ورکوي']]
```

### Chunking (Word Segmentation)

To retrieve full compound words instead of space-delimited tokens, use the Segmenter:

```python
from nlpashto import Segmenter

segmenter = Segmenter()
segmented_text = segmenter.segment(tokenized_text)
print(segmented_text)
# Output: [['جلال اباد', 'ښار', 'کې', 'هره', 'ورځ', 'لسګونه', 'کسان', 'په', 'ډله ییزه', 'توګه', 'د', 'نشه يي', 'توکو', 'کارولو', 'ته', 'ادامه', 'ورکوي']]
```

Specify batch size for multiple sentences:

```python
segmenter = Segmenter(batch_size=32)  # Default is 16
```

### Part-of-speech (POS) Tagging

For a detailed explanation about the POS tagger, refer to the [POS tagging paper](https://www.researchsquare.com/article/rs-2712906/v1):

```python
from nlpashto import POSTagger

pos_tagger = POSTagger()
pos_tagged = pos_tagger.tag(segmented_text)
print(pos_tagged)
# Output: [[('جلال اباد', 'NNP'), ('ښار', 'NNM'), ('کې', 'PT'), ('هره', 'JJ'), ('ورځ', 'NNF'), ...]]
```

### Sentiment Analysis (Offensive Language Detection)

Detect offensive language using a fine-tuned PsBERT model:

```python
from nlpashto import POLD

sentiment_analysis = POLD()

# Offensive example
offensive_text = 'مړه یو کس وی صرف ځان شرموی او یو ستا غوندے جاهل وی چې قوم او ملت شرموی'
sentiment = sentiment_analysis.predict(offensive_text)
print(sentiment)
# Output: 1

# Normal example
normal_text = 'تاسو رښتیا وایئ خور 🙏'
sentiment = sentiment_analysis.predict(normal_text)
print(sentiment)
# Output: 0
```

## Other Resources

### Pretrained Models

- **BERT (WordPiece Level):** [ijazulhaq/bert-base-pashto](https://huggingface.co/ijazulhaq/bert-base-pashto)
- **BERT (Character Level):** [ijazulhaq/bert-base-pashto-c](https://huggingface.co/ijazulhaq/bert-base-pashto-c)
- **Static Word Embeddings:** Available on [Kaggle](https://www.kaggle.com/datasets/drijaz/pashto-we): Word2Vec, fastText, GloVe

### Datasets and Examples

- Sample datasets: [Kaggle](https://www.kaggle.com/drijaz/)
- Jupyter Notebooks: [Kaggle](https://www.kaggle.com/drijaz/)

## Citations

**[NLPashto: NLP Toolkit for Low-resource Pashto Language](https://dx.doi.org/10.14569/IJACSA.2023.01406142)**

_H. Ijazul, Q. Weidong, G. Jie, and T. Peng, "NLPashto: NLP Toolkit for Low-resource Pashto Language," International Journal of Advanced Computer Science and Applications, vol. 14, no. 6, pp. 1345-1352, 2023._

- **BibTeX**

  ```bibtex
  @article{haq2023nlpashto,
    title={NLPashto: NLP Toolkit for Low-resource Pashto Language},
    author={Ijazul Haq and Weidong Qiu and Jie Guo and Peng Tang},
    journal={International Journal of Advanced Computer Science and Applications},
    issn={2156-5570},
    volume={14},
    number={6},
    pages={1345-1352},
    year={2023},
    doi={https://dx.doi.org/10.14569/IJACSA.2023.01406142}
  }
  ```

**[Correction of Whitespace and Word Segmentation in Noisy Pashto Text using CRF](https://doi.org/10.1016/j.specom.2023.102970)**

_H. Ijazul, Q. Weidong, G. Jie, and T. Peng, "Correction of whitespace and word segmentation in noisy Pashto text using CRF," Speech Communication, vol. 153, p. 102970, 2023._

- **BibTeX**

  ```bibtex
  @article{HAQ2023102970,
    title={Correction of whitespace and word segmentation in noisy Pashto text using CRF},
    journal={Speech Communication},
    issn={1872-7182},
    volume={153},
    pages={102970},
    year={2023},
    doi={https://doi.org/10.1016/j.specom.2023.102970},
    author={Ijazul Haq and Weidong Qiu and Jie Guo and Peng Tang}
  }
  ```

**[POS Tagging of Low-resource Pashto Language: Annotated Corpus and Bert-based Model](https://doi.org/10.21203/rs.3.rs-2712906/v1)**

_H. Ijazul, Q. Weidong, G. Jie, and T. Peng, "POS Tagging of Low-resource Pashto Language: Annotated Corpus and BERT-based Model," Preprint, 2023._

- **BibTeX**

  ```bibtex
  @article{haq2023pashto,
    title={POS Tagging of Low-resource Pashto Language: Annotated Corpus and Bert-based Model},
    author={Ijazul Haq and Weidong Qiu and Jie Guo and Peng Tang},
    journal={Preprint},
    year={2023},
    doi={https://doi.org/10.21203/rs.3.rs-2712906/v1}
  }
  ```

**[Pashto Offensive Language Detection: A Benchmark Dataset and Monolingual Pashto BERT](https://doi.org/10.7717/peerj-cs.1617)**

_H. Ijazul, Q. Weidong, G. Jie, and T. Peng, "Pashto Offensive Language Detection: A Benchmark Dataset and Monolingual Pashto BERT," PeerJ Computer Science, vol. 9, p. e1617, 2023._

- **BibTeX**

  ```bibtex
  @article{haq2023pold,
    title={Pashto Offensive Language Detection: A Benchmark Dataset and Monolingual Pashto BERT},
    author={Ijazul Haq and Weidong Qiu and Jie Guo and Peng Tang},
    journal={PeerJ Computer Science},
    issn={2376-5992},
    volume={9},
    pages={e1617},
    year={2023},
    doi={10.7717/peerj-cs.1617}
  }
  ```

**[Social Media’s Dark Secrets: A Propagation, Lexical and Psycholinguistic Ooriented Deep Learning Approach for Fake News Proliferation](https://doi.org/10.1016/j.eswa.2024.124650)**

_K. Ahmed, M. A. Khan, I. Haq, A. Al Mazroa, M. Syam, N. Innab, et al., "Social media’s dark secrets: A propagation, lexical and psycholinguistic oriented deep learning approach for fake news proliferation," Expert Systems with Applications, vol. 255, p. 124650, 2024._

- **BibTeX**

  ```bibtex
  @article{AHMED2024124650,
    title={Social media’s dark secrets: A propagation, lexical and psycholinguistic oriented deep learning approach for fake news proliferation},
    author={Kanwal Ahmed and Muhammad Asghar Khan and Ijazul Haq and Alanoud Al Mazroa and Syam M.S. and Nisreen Innab and Masoud Alajmi and Hend Khalid Alkahtani},
    journal={Expert Systems with Applications},
    volume={255},
    pages={124650},
    year={2024},
    issn={0957-4174},
    doi={https://doi.org/10.1016/j.eswa.2024.124650}
  }
  ```
  
## Contact
- Website: [https://ijaz.me/](https://ijaz.me/)
- LinkedIn: [https://www.linkedin.com/in/drijazulhaq/](https://www.linkedin.com/in/drijazulhaq/)
