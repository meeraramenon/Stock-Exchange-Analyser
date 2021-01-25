from urllib.request import urlopen
from urllib.request import Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

finviz_url = "https://finviz.com/quote.ashx?t="
tickers = ['AMZN', 'GOOG', 'FB']

news_tables = {} 

for ticker in tickers:
    url = finviz_url + ticker

    req = Request(url=url, headers={'user-agent': 'my-app'}) 
    response = urlopen(req)
    #print(response)  

    html = BeautifulSoup(response, features='html.parser') 
    #print(html)

    news_table = html.find(id='news-table') 
    news_tables[ticker] = news_table 
    #print(news_tables)
    
    #break


# amzn_data = news_tables['AMZN']
# amzn_rows = amzn_data.findAll('tr')
# print(amzn_rows)


# for index, row in enumerate(amzn_rows)
#     title = row.a.text #gets the text from anchor tag
#     timestamp = row.td.text #gets the text from td tag
#     print(timestamp + " " + title)


parsed_data = [] 

for ticker, news_table in news_tables.items():

    for row in news_table.findAll('tr'):

        title = row.a.text
        date_data = row.td.text.split(' ') 

        if len(date_data) == 1:
            time = date_data[0]
        else:
            date = date_data[0]
            time = date_data[1]

        parsed_data.append([ticker, date, time, title]) #array of array

#print(parsed_data)

#applying sentiment analysis part

df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title']) 

#print(df.head()) #print first 5 rows of pandas dataframe
#print(df["title"])

vader = SentimentIntensityAnalyzer()

#print(vader.polarity_scores("I love food. My personal favorite is idly"))

f = lambda title: vader.polarity_scores(title)['compound'] 
#can be written as
# def f(title):
#     return vader.polarity_scores(title)['compound']

# f(df['title'])

df['compound'] = df['title'].apply(f) 
#print(df.head())


df['date'] = pd.to_datetime(df.date).dt.date 

#visualizing part

plt.figure(figsize=(10,8)) 
mean_df = df.groupby(['ticker', 'date']).mean().unstack() 
#print(mean_df)


# mean_df = mean_df.unstack()
# mean_df = mean_df.xs('compound', axis="columns").transpose() 

mean_df = mean_df.xs('compound', axis="columns")
mean_df.plot(kind='bar')
plt.show()