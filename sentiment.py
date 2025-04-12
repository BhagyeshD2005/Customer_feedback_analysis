

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import nltk
import re
import warnings
warnings.filterwarnings('ignore')


def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            print(f"Downloading {resource}...")
            nltk.download(resource, quiet=False)
        except Exception as e:
            print(f"Error downloading {resource}: {e}")
            
download_nltk_resources()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

data = {
    'feedback': [
        "The product is amazing! I love it so much.",
        "Not satisfied with the quality, will not buy again.",
        "I had a great experience with customer service.",
        "Delivery was delayed and the product was damaged.",
        "Average product, nothing special about it.",
        "This is the best purchase I've made all year!",
        "Terrible experience, waste of money.",
        "The product meets my expectations.",
        "I'm disappointed with the features.",
        "Customer support was helpful and responsive.",
        "The app keeps crashing, very frustrating.",
        "Decent value for money, would recommend.",
        "Excellent quality and fast shipping!",
        "Too expensive for what it offers.",
        "Product works as advertised, no complaints.",
        "I regret buying this product.",
        "Absolutely love it! Will buy again.",
        "The website was confusing to navigate.",
        "Good product but shipping took too long.",
        "Not worth the price, look elsewhere.",
        "Packaging was beautiful and eco-friendly.",
        "Nothing extraordinary, just okay.",
        "Best customer service I've ever had!",
        "App interface is clean and easy to use.",
        "Refund process was a nightmare.",
        "Everything arrived on time and in perfect condition.",
        "Felt scammed, not what I expected at all.",
        "The design looks outdated.",
        "Happy with the product, does the job.",
        "Too many ads in the app, annoying.",
        "Product exceeded my expectations!",
        "Just average, wouldn’t recommend strongly.",
        "Broken item received, very disappointed.",
        "I’ve already recommended it to my friends!",
        "Slow loading time makes it unusable.",
        "Affordable and good build quality.",
        "Did not match the description.",
        "I’m really impressed by the performance.",
        "Installation instructions were confusing.",
        "Fair price, decent quality.",
        "Unhelpful and rude customer support.",
        "Loving the updates and new features!",
        "Not intuitive for new users.",
        "Sleek design and very durable.",
        "Service team resolved my issue quickly.",
        "Expected better for the price.",
        "Easy checkout process, hassle-free shopping.",
        "The color faded after one wash.",
        "Perfect gift, they absolutely loved it!",
        "Felt like a waste of my time and money."
    ],
    'sentiment': [
        'positive', 'negative', 'positive', 'negative', 'neutral',
        'positive', 'negative', 'neutral', 'negative', 'positive',
        'negative', 'positive', 'positive', 'negative', 'neutral',
        'negative', 'positive', 'negative', 'neutral', 'negative',
        'positive', 'neutral', 'positive', 'positive', 'negative',
        'positive', 'negative', 'neutral', 'positive', 'negative',
        'positive', 'neutral', 'negative', 'positive', 'negative',
        'positive', 'negative', 'positive', 'negative', 'neutral',
        'negative', 'positive', 'negative', 'positive', 'positive',
        'negative', 'positive', 'negative', 'positive', 'negative'
    ]
}


df = pd.DataFrame(data)

def preprocess_text(text):
    try:
        text = text.lower()
        
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        try:
            tokens = word_tokenize(text)
        except LookupError:
            tokens = text.split()
        
        try:
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words]
        except LookupError:
            pass
        
        try:
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
        except LookupError:
            pass
        
        return ' '.join(tokens)
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return text 

print("Preprocessing feedback...")
df['processed_feedback'] = df['feedback'].apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(
    df['processed_feedback'], 
    df['sentiment'], 
    test_size=0.3, 
    random_state=42
)

print("Training sentiment analysis model...")
sentiment_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000)),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

sentiment_pipeline.fit(X_train, y_train)

y_pred = sentiment_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(report)

def predict_sentiment(text):
    processed = preprocess_text(text)
    prediction = sentiment_pipeline.predict([processed])[0]
    probabilities = sentiment_pipeline.predict_proba([processed])[0]
    confidence = max(probabilities)
    return prediction, confidence

sample_texts = [
    "I hate this product! Will definitely buy again. (sarcasm)",
    "The quality is not what I expected. Really disappointed and won’t recommend it.",
    "It's okay, works as expected. Nothing great, nothing terrible.",
    "Absolutely thrilled with this purchase! I use it every day and it hasn’t let me down.",
    "Terrible product. Broke within a week and the company refused to refund me.",
    "I had my doubts at first, but this product really surprised me—in a good way!",
    "Meh. It does the job but don't expect too much.",
    "Customer service was very helpful and resolved my issue within minutes.",
    "It’s overpriced and underperforms. Definitely not worth the hype.",
    "I love the packaging and design, but the actual product didn’t live up to my expectations.",
    "Wow! This actually made my life easier. Super happy with the purchase.",
    "I’ve had better. It’s usable, but I won’t be buying it again.",
    "What a scam! Totally not what was advertised.",
    "Worked like a charm. Installation was simple and results were excellent.",
    "The app is a total mess—lags, bugs, and crashes constantly.",
    "Fast delivery, decent performance, and good customer support.",
    "Didn't love it, didn’t hate it. Just a standard experience.",
    "They sent the wrong item, and now I’m stuck with it. Extremely frustrating.",
    "Quality seems fine, but let’s see how it holds up over time.",
    "Not bad at all. In fact, I’m thinking of buying another one."
]

print("\nSample Predictions:")
for text in sample_texts:
    sentiment, confidence = predict_sentiment(text)
    print(f"Text: {text}")
    print(f"Predicted sentiment: {sentiment} (confidence: {confidence:.2f})")
    print()

sentiment_counts = df['sentiment'].value_counts()

print("Creating visualizations...")
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=['green', 'gray', 'red'])
plt.title('Overall Sentiment Distribution')

plt.subplot(2, 2, 2)
sns.countplot(x='sentiment', data=df, palette={'positive': 'green', 'neutral': 'gray', 'negative': 'red'})
plt.title('Sentiment Count')
plt.xlabel('Sentiment')
plt.ylabel('Count')

plt.subplot(2, 2, 3)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=sentiment_pipeline.classes_, 
            yticklabels=sentiment_pipeline.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(2, 2, 4)
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame({
    'Precision': [report_dict[c]['precision'] for c in ['positive', 'neutral', 'negative']],
    'Recall': [report_dict[c]['recall'] for c in ['positive', 'neutral', 'negative']],
    'F1-Score': [report_dict[c]['f1-score'] for c in ['positive', 'neutral', 'negative']]
}, index=['Positive', 'Neutral', 'Negative'])
report_df.plot(kind='bar', colormap='viridis')
plt.title('Model Performance Metrics')
plt.xlabel('Sentiment')
plt.ylabel('Score')
plt.legend(loc='lower right')
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig('sentiment_analysis_results.png')
plt.show()

try:
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=90, freq='D')
    daily_feedback_count = np.random.randint(5, 30, size=90)
    positive_ratio = 0.5 + 0.3 * np.sin(np.linspace(0, 3, 90)) + np.random.normal(0, 0.1, 90)
    positive_ratio = np.clip(positive_ratio, 0, 1)

    time_df = pd.DataFrame({
        'date': dates,
        'feedback_count': daily_feedback_count,
        'positive_ratio': positive_ratio,
        'negative_ratio': 1 - positive_ratio
    })

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time_df['date'], time_df['feedback_count'], 'b-')
    plt.title('Daily Feedback Volume')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(time_df['date'], time_df['positive_ratio'], 'g-', label='Positive')
    plt.plot(time_df['date'], time_df['negative_ratio'], 'r-', label='Negative')
    plt.title('Sentiment Trend Over Time')
    plt.ylabel('Ratio')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig('sentiment_trend_analysis.png')
    plt.show()
except Exception as e:
    print(f"Skipping time series visualization due to error: {e}")

try:
    from wordcloud import WordCloud
    
    # Join all positive and negative feedback
    positive_text = ' '.join(df[df['sentiment'] == 'positive']['processed_feedback'])
    negative_text = ' '.join(df[df['sentiment'] == 'negative']['processed_feedback'])

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    if positive_text.strip():
        wordcloud_positive = WordCloud(width=400, height=200, background_color='white', max_words=100, colormap='Greens').generate(positive_text)
        plt.imshow(wordcloud_positive, interpolation='bilinear')
    plt.title('Positive Feedback Word Cloud')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    if negative_text.strip():
        wordcloud_negative = WordCloud(width=400, height=200, background_color='white', max_words=100, colormap='Reds').generate(negative_text)
        plt.imshow(wordcloud_negative, interpolation='bilinear')
    plt.title('Negative Feedback Word Cloud')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('sentiment_wordclouds.png')
    plt.show()
except ImportError:
    print("WordCloud package not available. Skipping word cloud visualization.")
except Exception as e:
    print(f"Error creating word clouds: {e}")

print("Sentiment analysis and visualization complete!")
