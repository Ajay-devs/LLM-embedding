# 🐦 Twitter Sentiment Classification Using Text Embeddings

A machine learning project that classifies tweet sentiments (Positive, Negative, Neutral) using **Sentence Transformer embeddings** and multiple ML classifiers.

## 📊 Results

| Model | Accuracy | F1 Score |
|---|---|---|
| **XGBoost** 🏆 | **84.88%** | **84.88%** |
| Logistic Regression | 84.15% | 84.16% |
| Random Forest | 80.04% | 79.91% |

## 🚀 Pipeline

1. **Data Loading** — 27,476 tweets with sentiment labels
2. **EDA** — Sentiment distribution, word clouds, text length analysis
3. **Text Preprocessing** — URL/mention removal, lowercasing, special character cleanup
4. **Embedding Generation** — `all-MiniLM-L6-v2` Sentence Transformer (384-dim vectors)
5. **Model Training** — Logistic Regression, Random Forest, XGBoost
6. **Evaluation** — Classification reports, confusion matrices, accuracy comparison
7. **Custom Predictions** — 5 test tweets classified with confidence scores

## 🛠️ Tech Stack

- **Embeddings**: [Sentence Transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`)
- **ML Models**: Scikit-learn, XGBoost
- **Visualization**: Matplotlib, Seaborn, WordCloud
- **Data**: Pandas, NumPy

## 📂 Project Structure

```
├── Sentiment_Classification_Embeddings.ipynb   # Main notebook (with outputs)
├── .gitignore
└── README.md
```

## ▶️ How to Run

1. Open `Sentiment_Classification_Embeddings.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Click **Runtime → Run All**
3. All required libraries are installed automatically in the first cell

## 📋 Dataset

- **Source**: Twitter Sentiment Extraction dataset
- **Size**: 27,476 tweets
- **Classes**: Positive (8,582), Neutral (11,118), Negative (7,781)
- **Columns**: `textID`, `text`, `selected_text`, `sentiment`

## ✅ Features

- ✅ Sentence Transformer text embeddings
- ✅ Sentiment distribution bar chart
- ✅ Class-wise word clouds (Positive / Neutral / Negative)
- ✅ Text length analysis with histograms & box plots
- ✅ Classification reports for all 3 models
- ✅ Confusion matrices with heatmaps + interpretation
- ✅ Custom tweet predictions with confidence scores

## 📝 License

This project is for educational purposes as part of the NIAT Masterclass.
