import nltk
nltk.download('punkt_tab')
import emoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Download required data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize analyzer
analyzer = SentimentIntensityAnalyzer()

# ----------------------------------
# 1. Sentiment Analysis
# ----------------------------------
def text_sentiment_analysis(text):
    vader_score = analyzer.polarity_scores(text)
    polarity = TextBlob(text).sentiment.polarity

    if vader_score['compound'] >= 0.05:
        sentiment = "Positive"
    elif vader_score['compound'] <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return sentiment, vader_score, polarity

# Example
text = "The product quality is amazing 😊"
print(text_sentiment_analysis(text))


# ----------------------------------
# 2. Topic Modeling (LDA)
# ----------------------------------
from gensim import corpora, models
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

documents = [
    "Battery problem in mobile phone",
    "Camera quality is excellent",
    "Poor delivery service",
    "Battery drains fast issue"
]

texts = [
    [word for word in word_tokenize(doc.lower())
     if word.isalpha() and word not in stop_words]
    for doc in documents
]

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

lda = models.LdaModel(
    corpus=corpus,
    num_topics=2,
    id2word=dictionary,
    passes=10
)

print("\nLDA Topics:")
print(lda.print_topics())


# ----------------------------------
# 3. Emoji Sentiment
# ----------------------------------
def emoji_sentiment(text):
    emojis = [char for char in text if char in emoji.EMOJI_DATA]

    positive = ['😊', '😍', '👍', '😁', '🎉']
    negative = ['😡', '😠', '😢', '👎']

    score = 0

    for e in emojis:
        if e in positive:
            score += 1
        elif e in negative:
            score -= 1

    if score > 0:
        return "Positive Emoji Sentiment"
    elif score < 0:
        return "Negative Emoji Sentiment"
    else:
        return "Neutral Emoji Sentiment"

# Example
print(emoji_sentiment("Worst service ever 😡"))


# ----------------------------------
# 4. Business Decision
# ----------------------------------
def business_decision(sentiment, emoji_sent):
    if sentiment == "Negative" or "Negative" in emoji_sent:
        return "Immediate Action Required"
    elif sentiment == "Positive":
        return "Campaign Successful"
    else:
        return "Monitor Feedback"

# Example
print(business_decision("Negative", "Negative Emoji Sentiment"))