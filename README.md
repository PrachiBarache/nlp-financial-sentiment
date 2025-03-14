
# Financial Sentiment Analysis

This repository contains the implementation of a financial sentiment analysis system that classifies financial news statements as positive, negative, or neutral. The project compares the performance of transformer-based models (BERT and RoBERTa) against traditional machine learning methods (SVM) for sentiment classification of financial texts.

## Problem Statement

The primary goal is to perform financial sentiment analysis using financial news statements. Financial news can be a source of new information, expressed as sentiment, which has proven useful in enhancing stock market prediction with statistical and machine learning approaches. Sentiment analysis of financial news helps to understand public sentiment, timely access to public opinions and attitudes, and rapidly extract the subject of information.

## Hypothesis

Transformer-based models, specifically BERT and RoBERTa, will deliver superior accuracy in sentiment classification of financial news texts compared to traditional methods such as SVM. The advanced capability of BERT and RoBERTa to capture wider and deeper contextual relationships between words enables them to identify subtle nuances in financial contexts, leading to more precise sentiment determinations.

## Dataset

The [Financial PhraseBank dataset](https://huggingface.co/datasets/financial_phrasebank) consists of approximately 4,850 sentences from English financial news, each annotated as "Positive", "Negative", or "Neutral" by experts with backgrounds in finance. This dataset is specifically tailored for text classification tasks focused on sentiment analysis within the financial sector.

Examples from the dataset:
- "According to Gran, the company has no plans to move all production to Russia, although that is where the company is growing." - Neutral
- "The international electronic industry company Elcoteq has laid off tens of employees from its Tallinn facility." - Negative
- "With the new production plant the company would increase its capacity..." - Positive

## Models Implemented

1. **Support Vector Machine (SVM)**: Selected as a baseline model for its capability in high-dimensional spaces and robustness against overfitting.

2. **Recurrent Neural Network (RNN)**: Used to capture sequential context, important for understanding sentiments in texts.

3. **BERT (Bidirectional Encoder Representations from Transformers)**: Provides superior context understanding through its attention mechanisms and pre-trained contextual embeddings.

4. **RoBERTa (Robustly Optimized BERT Approach)**: An enhanced version of BERT with optimized training regimes.

## Results

| Model | Accuracy | Precision (Avg) | Recall (Avg) | F1-Score (Avg) |
|-------|----------|----------------|--------------|----------------|
| SVM   | 83%      | 0.81           | 0.73         | 0.76           |
| RNN   | 71%      | -              | -            | -              |
| BERT  | 92%      | 0.89           | 0.92         | 0.90           |
| RoBERTa | 95%    | 0.92           | 0.92         | 0.92           |

The results confirm the hypothesis that transformer-based models (BERT and RoBERTa) outperform traditional machine learning methods (SVM and RNN) in financial sentiment analysis. RoBERTa achieved the highest performance with 95% accuracy.

## Project Structure

```
financial-sentiment-analysis/
├── data/
│   └── financial_phrasebank.csv
├── models/
│   ├── svm_model.py
│   ├── rnn_model.py
│   ├── bert_model.py
│   └── roberta_model.py
├── preprocessing/
│   └── text_preprocessing.py
├── evaluation/
│   └── model_evaluation.py
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│   └── results_visualization.ipynb
├── requirements.txt
└── README.md
```

## Dependencies

- Python 3.8+
- TensorFlow 2.x
- PyTorch 1.x
- Transformers (Hugging Face)
- Scikit-learn
- NLTK
- Pandas
- Matplotlib

## Conclusion

The project successfully implemented and compared different approaches for financial sentiment analysis. Transformer-based models (BERT and RoBERTa) demonstrated significantly better performance than traditional machine learning methods (SVM and RNN) due to their advanced contextual understanding capabilities. The use of pre-trained contextual embeddings proved crucial in capturing the nuanced sentiments expressed in financial texts.
