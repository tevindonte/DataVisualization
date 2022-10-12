import nest_asyncio
nest_asyncio.apply()
import pandas
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from nltk import bigrams
import twint
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from afinn import Afinn
from nltk.corpus import sentiwordnet as swn
from wordcloud import WordCloud
import csv



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

af = Afinn(language='en') # Words scores range from minus five (negative) to plus five (positive).
af = Afinn(emoticons=True) 
df = pandas.read_csv ('Data.csv')
tweetcount=len(df)-1
tweet = np.array(df['tweet']) #Create an array that has all times
time = np.array(df['time']) #Create an array that has all times
test_tweet = tweet[0:tweetcount+1] # extract data for model evaluation
test_time = time[0:tweetcount+1]

sentiment_polarity = [af.score(tweet) for tweet in test_tweet] #Apply Scores
possent=[] #Range of values that are positive separated
negsent=[] #Range of values that are negative separated
for item in sentiment_polarity:
    if item < 0:
        negsent.append(item)
    else:
        possent.append(item)

np.savetxt("pos.csv", possent, delimiter=", ", fmt="% s", header='Positive') #Creates csv of Positives
np.savetxt("neg.csv", negsent, delimiter=", ", fmt="% s", header='Negative') #Creates csv of Negatives


dfpos=pandas.read_csv ('pos.csv')
dfneg=pandas.read_csv ('neg.csv')

dfconcat=pandas.concat([dfpos,dfneg]) #Combines both
nulls = {'NULL', 'null', 'None', ''}
#dfconcat.to_csv('bothconcat.csv', index=False) #Turns it into csv
dfboth=pandas.read_csv('bothconcat.csv')
poscount=(len(dfpos))
negcount=(len(dfneg))
dfboth['# Negative'] = dfboth['# Negative'].fillna(0)
dfboth['# Positive'] = dfboth['# Positive'].fillna(0)
#dfboth.to_csv('nanconcat.csv', index=False) #Turns it into csv
labels = ['Positive', 'Negative'] #Axis labels
data = [poscount, negcount] #Axis
fig = plt.figure(figsize =(10, 7))
plt.pie(data, labels = labels) #Matplot Pie Graph
# show plot
plt.show()
print("The number of positive tweets: ", poscount)
print("The number of negative tweets: ", negcount)


predicted_sentiments = ['positive' if score >= 1.0 else 'negative' for score in sentiment_polarity] #Change Scores to negative and positive
poscount = predicted_sentiments.count('positive') #Count positive tweets
negcount = predicted_sentiments.count('negative') #Count negative tweets
bothcount=poscount+negcount

tweetslist = df['tweet'].tolist() #Turn tweets to list
timelist = df['time'].tolist() #Turn time to list
rangerp = dfpos['# Positive'].tolist() #Turn tweets to list
rangern = dfneg['# Negative'].tolist() #Turn tweets to list

rangerp.plot(kind='bar')


unique_string=(" ").join(tweetslist) #WordCloud Setup
wordcloud = WordCloud(width = 1000, height = 500).generate(unique_string)
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig("your_file_name"+".png", bbox_inches='tight')
plt.show()
plt.close()


data2=poscount+negcount

plt.plot(dfpos, color='magenta', marker='o',mfc='pink' ) #plot the data
plt.xticks(range(0,data2)) #set the tick frequency on x-axis
plt.ylabel('frequency') #set the label for y axis
plt.xlabel('number of tweets') #set the label for x-axis
plt.title("Positive Frequency") #set the title of the graph

plt.plot(dfneg, color='red', marker='o',mfc='pink' ) #plot the data
plt.xticks(range(0,data2)) #set the tick frequency on x-axis
plt.ylabel('frequency') #set the label for y axis
plt.xlabel('number of tweets') #set the label for x-axis
plt.title("Negative Frequency") #set the title of the graph
print('The number of tweets :', data2)
plt.show() #display the graph



