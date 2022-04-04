import pandas as pd

df = pd.read_csv("twitter_sentiment_data.csv")
df.sentiment.value_counts()
df['encoded_cat'] = df.sentiment.astype("category").cat.codes
df = df[df['message'] != None]
df.dropna(inplace=True)
df = df.drop_duplicates(subset=['message'],keep='first')

# Read indices of conflicted pairs
conflicted = []
f = open('unsimilar_pairs.txt', encoding='utf-8')
lines = f.readlines()
for line in lines:
    a, b = line.split()
    conflicted.append(int(a))
    conflicted.append(int(b))
    # # Output for checking
    # print("a=",a,"b=",b)
    # print(df.loc[a]['message'], df.loc[a]['sentiment'])
    # print(df.loc[b]['message'], df.loc[b]['sentiment'])
    # print('-'*30)

conflicted = list(set(conflicted))

# Remove conflicted tweets
df = df.drop(labels=conflicted, axis=0)
print(len(df), ' tweets after cleaning.')

# Save dataframe to csv, index for line index, default = True
df.to_csv("twitter_sentiment_data_clean.csv", columns=['sentiment', 'message', 'tweetid'], \
          index=False, sep=',', encoding='utf-8')
