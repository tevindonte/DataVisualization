import nest_asyncio
nest_asyncio.apply()
import pandas
import numpy as np
import matplotlib.pyplot as plt
import twint
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from afinn import Afinn





print ("Fetching Tweets")
config = twint.Config()
username=input("What is the username? Type No If not" )
if(username!='No'):
    config.Username=username #Username of twitter account
text = input('Enter text to search for: ')
config.Search = text #Search terms
lang = input('Choose language (en - ar - ...) : ')
config.Lang = lang #Compatible language
tlimit=int(input('What is your tweet limit?'))
config.Limit=(tlimit) #Number of Tweets to pull
config.Store_csv=True #Set to True to write as a csv file.
config.Custom_csv = ["id", "user_id", "username", "tweet"]
config.Output='Data.csv' #Name of the output file.
twint.run.Search(config)
print('Scraping Done!')
config.Pandas = True


afinn = Afinn(language='en') # Words scores range from minus five (negative) to plus five (positive).
df = pandas.read_csv ('Data.csv')
tweetcount=len(df)-1
df['afinn_score'] = df['tweet'].apply(afinn.score)
def splitter(text_string):     #Calculate the number of words in a string
    return len(text_string.split())
df['splitter'] = df['tweet'].apply(splitter)
#df['word_count'].describe()
#df['afinn_score'].describe()
columns_to_display = ['tweet', 'afinn_score']
pandas.set_option('max_colwidth', 100)
#df.sort_values(by='afinn_score')[columns_to_display].head(10)
df['afinn_adjusted'] = df['afinn_score'] / df['splitter'] * 100 # to make the results more readable, the adjustment is multiplied by 100.


analyzer = SentimentIntensityAnalyzer()
sentiment = df['tweet'].apply(analyzer.polarity_scores)
sent = pandas.DataFrame(sentiment.tolist())
def vaderize(df, textfield):
    '''Compute the Vader polarity scores for a textfield.
    Returns scores and original dataframe.'''

    analyzer = SentimentIntensityAnalyzer()

    print('Estimating polarity scores for %d cases.' % len(df))
    sentiment = df[textfield].apply(analyzer.polarity_scores)

    # convert to dataframe
    sdf = pandas.DataFrame(sentiment.tolist()).add_prefix('vader_')

    # merge dataframes
    df_combined = pandas.concat([df,sdf], axis = 1)
    return df_combined
vaderized = vaderize(df, 'tweet')
sentiment_variables = ['afinn_adjusted', 'vader_neg', 'vader_neu', 'vader_pos']
vaderized['vader_compound'].plot(kind='hist')
vaderized.plot.scatter(x='vader_pos', y = 'vader_neg')

def en(x):
    if x =='en':
        return 'English'
    else:
        return 'Other'
tweeter=df.loc[:,"tweet"]

tweeter['en']=tweeter['language'].apply(lambda x: en(x))
ax = tweeter['en'].value_counts().plot.pie(autopct='%1.1f%%',title='Share of English in All Languages')
ax.set_ylabel('')

df_lan = tweeter[tweeter['language'] != 'ja']
df_lan['language'].value_counts().plot(kind='bar')
