from urllib.request import urlopen
from urllib.request import Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

finviz_url = "https://finviz.com/quote.ashx?t="
tickers = ['AMZN', 'GOOG', 'FB']

news_tables = {} #dictionary, stores the heading artcles insdie that table on webpage

for ticker in tickers:
    url = finviz_url + ticker

    req = Request(url=url, headers={'user-agent': 'my-app'}) #specify url param, header allows us to access data, without user-agent , forbid access, cannot downlaod data from finviz, im juts accesing website from this user agent, so now ima name is my-app
    response = urlopen(req)
    #print(response) 
    # -> will show the http client http response oj=bject shwoing connection has been established
    #take this response and throw it into soup

    html = BeautifulSoup(response, features='html.parser') #till 17 we have the data from website (webscraped) and ready to be parsed. we pass the http response we just got and mention its an html parser that we wanna use
    #print(html)
    # -> whatever we saw on inspect element will see through print statemnt, parsing html code from the url
    #all our needed content is inside a table id=news-table

    news_table = html.find(id='news-table') #gets us the html object of the entire table, with all different results
    news_tables[ticker] = news_table #key will be ticker value
    #basically we take the table object from html and store it in a dictionary
    #we can parse em now itself, but easier to understand if we first save it to dict and then parse, inefficient tho
    #print(news_tables) -> shows the table data from html code
    #now we parse data and get it to an understandable format so that we can extract title, timestamp and then eventually apply sentiment analys

    #basically iterate through each row, get timestamp, href and most importantly the title or text of the article

    #break


#gets the <tr> data only and prints it in a list like fashion, comma
# amzn_data = news_tables['AMZN']
# amzn_rows = amzn_data.findAll('tr')
# print(amzn_rows)


#iterate through table rows

# for index, row in enumerate(amzn_rows): #enumerate funct gives index and object of any list item or list and iterates through every single object 
#     title = row.a.text #gets the text from anchor tag
#     timestamp = row.td.text #gets the text from td tag
#     print(timestamp + " " + title)

#     #now we have to convert this in such a way that it works for any ticker, right now its only test focused
#     #so now we iterate throught the news_tables that has all the tickers that we are going to populate, then we will scrape the data and add it to a new list of items

parsed_data = [] #here we are making a list or a list within a list that has the title, timestamp and ticker (coress)

for ticker, news_table in news_tables.items(): #iterating over the key and value pairs in the news_tables dict

    for row in news_table.findAll('tr'):

        title = row.a.text
        date_data = row.td.text.split(' ') #split to sections at space
        #if the length of date_data > 1, then we know that there is date and time else only time 

        if len(date_data) == 1:
            time = date_data[0]
        else:
            date = date_data[0]
            time = date_data[1]

        parsed_data.append([ticker, date, time, title]) #array of array

#print(parsed_data)

#applying sentiment analysis part
#our aim is to apply the function form nltk's module to the parsed data

df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title']) #create a dataframe to host our data in

#print(df.head()) #print first 5 rows of pandas dataframe
#print(df["title"])

vader = SentimentIntensityAnalyzer()
#vader is the sentiment intensity analyser
#analyses any given text

#print(vader.polarity_scores("I love food. My personal favorite is idly"))

f = lambda title: vader.polarity_scores(title)['compound'] #the funct takes in title, then it says give me polarity score of title, but i only need compund
#so whatever string i pass into this function, i want u to just gimme back compound

#can be written as
# def f(title):
#     return vader.polarity_scores(title)['compound']

# f(df['title'])

df['compound'] = df['title'].apply(f) #take every single title and apply a function which is a lambda funct
#print(df.head())

#date is string right now
#convert to recognisable date time format so we have a hiearchy of what date came when
df['date'] = pd.to_datetime(df.date).dt.date #we are modifying the date column
#hence when we visualize in matplotlib and draw the chart, we will make sure the months of may june etc, they will come in correct order

#visualizing part

plt.figure(figsize=(10,8)) #set figure size
#find avg of all compound values and check was today a good or bad day for amazon
mean_df = df.groupby(['ticker', 'date']).mean().unstack() #groups data by ticker and date, this code gets the mean/avg of all compund values for that ticker and date -> hence we get only one value -> only one date entry
#print(mean_df)

#by using unstack, it allows us to have date as x axis
# mean_df = mean_df.unstack()
# mean_df = mean_df.xs('compound', axis="columns").transpose() #shows crossection of compound, basically shows data in a neat way
# xs ->taking a crossection, crossection will be of compound, specify axis as columns and then transpose this data
#basically manipulating the dataframe, transposing it, so that we get key value pairs of the dates and compound values

mean_df = mean_df.xs('compound', axis="columns")
mean_df.plot(kind='bar')
plt.show()