import pandas as pd
data = pd.read_csv("new sma_tools_techniques_engagement_metrics.csv")
data["Engagement_Rate"] = ((data["Likes"] + data["Comments"] + data["Shares"]) /
data["Followers"]) * 100
# Sort by Engagement Rate and display top posts
top_posts = data.sort_values(by="Engagement_Rate", ascending=False)
print(top_posts.head())
import matplotlib.pyplot as plt
plt.bar(data["Post Date"], data["Engagement_Rate"])
plt.xticks(rotation=45)
plt.title("Engagement Rate per Post")
plt.xlabel("Post Date")
plt.ylabel("Engagement Rate (%)")
plt.show()
