
# 🧠 Sentimental Data Analysis Model- Customer Feedback Analysis

This repository demonstrates a complete **Natural Language Processing (NLP)** pipeline for sentiment analysis using an **in-memory dataset**. It utilizes Python, machine learning (Logistic Regression), and text processing tools (NLTK, TF-IDF) to classify user feedback as **positive**, **neutral**, or **negative**. The project includes comprehensive **data visualizations** to provide insights into model performance and sentiment trends.

---

## 📌 Key Features

- ✅ **Self-contained script** with no external data files — all feedback is stored in memory.
- 🔍 Text preprocessing using **NLTK**: tokenization, stopword removal, and lemmatization.
- 🤖 Sentiment classification with a **TF-IDF Vectorizer** and **Logistic Regression**.
- 📊 Evaluation via **accuracy**, **precision**, **recall**, **F1-score**, and **confusion matrix**.
- 📈 Generation of insightful visualizations:
  - Sentiment distribution (pie chart and bar chart)
  - Confusion matrix heatmap
  - Classification metrics bar chart
  - Simulated time-based sentiment trends
  - Word clouds for positive and negative feedback
- 🔮 Prediction functionality with confidence scores for new input texts

---

## 📂 Project Structure

```bash
.
├── sentiment_analysis.py             # Main script with NLP pipeline and visualizations
├── sentiment_analysis_results.png    # Sentiment breakdown and performance metrics
├── sentiment_trend_analysis.png      # Simulated daily sentiment trends
├── sentiment_wordclouds.png          # Word clouds for sentiment categories
├── README.md                         # Project overview and documentation
```

---

## 📦 Requirements

Install the required libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud
```

The script automatically downloads necessary NLTK resources:

```python
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## 🧠 Dataset Overview

This project uses a manually curated dataset embedded directly within the Python script. Each feedback entry is labeled with a sentiment:

```python
data = {
    'feedback': [ ... ],    # List of user reviews
    'sentiment': [ ... ]    # Corresponding labels: 'positive', 'neutral', or 'negative'
}
```

---

## 🚀 How to Run the Project

To execute the entire pipeline and generate visualizations:

```bash
python sentiment_analysis.py
```

This will:
- Preprocess and clean the text
- Train a machine learning model
- Evaluate performance on a test split
- Generate and save visualizations
- Display sample predictions with confidence levels

---

## 📊 Model Evaluation

The model's performance is assessed using:

- **Accuracy Score**
- **Classification Report**
  - Precision, Recall, and F1-Score for each class
- **Confusion Matrix**

---

## 🖼️ Visual Output Samples

### 1. Sentiment Analysis Results

- Pie and bar charts for sentiment distribution
- Confusion matrix and classification metrics

📁 Output: `sentiment_analysis_results.png`

---

### 2. Time-Based Sentiment Trends (Simulated)

- Feedback volume over time
- Positive vs. negative sentiment trends

📁 Output: `sentiment_trend_analysis.png`

---

### 3. Word Clouds

- Most frequent terms in positive and negative feedback

📁 Output: `sentiment_wordclouds.png`

---

## 🧪 Sample Prediction Output

The script includes sample texts to test the trained model. Example output:

```text
Text: Terrible product. Broke within a week and the company refused to refund me.
Predicted sentiment: negative (confidence: 0.94)
```

You can modify the sample texts or call the `predict_sentiment()` function with your own inputs.

---

## 🔄 Future Enhancements

- Support for sarcasm and irony detection
- Incorporation of deep learning models (e.g., LSTM, BERT)
- Real-time feedback ingestion via API or web form
- Deployment via a Streamlit or Flask web app

---

## 📄 License

You are free to use, modify, and distribute it as needed.

---

## 🙏 Acknowledgements

This project utilizes the following open-source libraries:

- [NLTK](https://www.nltk.org/) – for natural language processing
- [Scikit-learn](https://scikit-learn.org/) – for model training and evaluation
- [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) – for data visualization
- [WordCloud](https://github.com/amueller/word_cloud) – for visualizing common terms

---

> Crafted with 💡 and 💬 for data scientists, developers, and NLP enthusiasts.

---

