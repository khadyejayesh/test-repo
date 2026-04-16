import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from wordcloud import WordCloud

# ----------------------------------
# STEP 1 : Load Twitter Dataset
# ----------------------------------
url = "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv"

df = pd.read_csv(url)

# Take small sample
df = df.head(500)

# Rename column
df = df.rename(columns={"tweet": "Tweet"})

# Add dummy date column
df["Date"] = pd.date_range(start="2024-01-01", periods=len(df))

# ----------------------------------
# STEP 2 : Text Cleaning
# ----------------------------------
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    return text.lower()

df["Clean_Tweet"] = df["Tweet"].apply(clean_text)

# ----------------------------------
# STEP 3 : Date Processing
# ----------------------------------
df["Date"] = pd.to_datetime(df["Date"])
df["Date_only"] = df["Date"].dt.date

# ----------------------------------
# STEP 4 : Tweets Over Time
# ----------------------------------
plt.figure(figsize=(10,5))
df["Date_only"].value_counts().sort_index().plot(kind="line")
plt.title("Tweets Over Time")
plt.xlabel("Date")
plt.ylabel("Tweet Count")
plt.grid()
plt.show()

# ----------------------------------
# STEP 5 : Likes vs Retweets (Dummy)
# ----------------------------------
df["Likes"] = range(1, len(df)+1)
df["Retweets"] = df["Likes"] // 2

plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="Likes", y="Retweets")
plt.title("Likes vs Retweets")
plt.grid()
plt.show()

# ----------------------------------
# STEP 6 : WordCloud
# ----------------------------------
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(" ".join(df["Clean_Tweet"]))

plt.figure(figsize=(10,5))
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Word Cloud")
plt.show()