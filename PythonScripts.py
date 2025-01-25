import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import nltk
import string


### Column explaination ###
# The column or features in the dataset:
# Id
# ProductId — unique identifier for the product
# UserId — unqiue identifier for the user
# ProfileName
# HelpfulnessNumerator — number of users who found the review helpful
# HelpfulnessDenominator — number of users who indicated whether they found the review helpful or not
# Score — rating between 1 and 5
# Time — timestamp for the review
# Summary — brief summary of the review
# Text — text of the review

# Create a SQL connection to our SQLite database
con = sqlite3.connect(r'C:\Users\30694\OneDrive\Έγγραφα\Project Portofolio\Amazon Customer Data Analysis/database.sqlite')

type(con) # you can see the type of the connection

#### reading data from Sqlite database
df=pd.read_sql_query("SELECT * FROM Reviews", con)
df.shape # to see the shape of the dataframe

## read only the first three rows
pd.read_sql_query("SELECT * FROM Reviews LIMIT 3", con)


## Read the dataset from an excel file
df = pd.read_csv(r'C:\Users\30694\OneDrive\Έγγραφα\Project Portofolio\Amazon Customer Data Analysis/Reviews.csv')

print(df.shape)
df.head()



######## ---------------- PERFORM SENTIMENT ANALYSIS  ----------------########
# ### What is sentiment analysis?
#     Sentiment analysis is the computational task of automatically determining what feelings a writer is expressing in text
#     Some examples of applications for sentiment analysis include:

#     1.Analyzing the social media discussion around a certain topic
#     2.Evaluating survey responses
#     3.Determining whether product reviews are positive or negative

#     Sentiment analysis is not perfect.It also cannot tell you why a writer is feeling a certain way. However, it can be useful to quickly summarize some qualities of text, especially if you have so much text that a human reader cannot analyze it.For this project,the goal is to to classify Food reviews based on customers' text.

from textblob import TextBlob
df['Summary'][0]
TextBlob(df['Summary'][0]).sentiment.polarity #it can take values from -1 to 1 and 0 indicates neutral , 1 positive polarity
# we got 0.7 that means it seems  to have a positive sentiment

## takes 3 mins 
polarity=[] # list which will contain the polarity of the comments , it is blank so that i will store it after

for i in df['Summary']:
    try:
        polarity.append(TextBlob(i).sentiment.polarity) #wtv poolarity i get from the parenthesis i go and store it to the blank list  
    except:
        polarity.append(0) # we want it for the null values
        
len(polarity) # from the result we understand how much complex data we have

data=df.copy()
data['polarity']=polarity # create a new column with the polarity at the new datarame

data['polarity'].nunique()




######## ---------------- PERFORM EDA POSITIVE SENTENCES  ----------------########

# we mean we will only analyse with polarity bigger than 0
data_positive = data[data['polarity']>0]
data_positive.shape

from wordcloud import WordCloud, STOPWORDS
stopwords=set(STOPWORDS)
positive=data_positive[0:200000]

total_text= (' '.join(data_positive['Summary']))
len(total_text)

total_text[0:10000] # this is a sample of the first 10000 values of the summary that has positive text

import re # if i see the above sample, i see some dots or special characters
# the above library helps to do modifications to text data 
total_text=re.sub('[^a-zA-Z]',' ',total_text) #replace special characters with space
total_text

## remove extra spaces
total_text=re.sub(' +',' ',total_text)
total_text[0:10000]
len(total_text)

wordcloud = WordCloud(width = 1000, height = 500,stopwords=stopwords).generate(total_text)
plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off') # it removes the axis

######## ---------------- PERFORM EDA NEGATIVE SENTENCES  ----------------########

# the same as the above but with <0

## --------------- Analyse to what User Amazon Can recommend more product ----------------------##

df['UserId'].shape
df['UserId'].nunique() # number of unique user

raw=df.groupby(['UserId']).agg({'Summary':'count', 'Text':'count','Score':'mean','ProductId':'count'}).sort_values(by='Text',ascending=False)
raw # groupping the data per user and counts the summary, the text etc


raw.columns=['Number_of_summaries','num_text','Avg_score','Number_of_products_purchased']
raw # rename the columns

user_10=raw.index[0:10]
number_10=raw['Number_of_products_purchased'][0:10]

plt.bar(user_10, number_10, label='java developer')
plt.xlabel('User_Id')
plt.ylabel('Number of Products Purchased')
plt.xticks(rotation='vertical') # rotates the values of the x axis
# These are the Top 10 Users so we can recommend more & more Prodcuts to these Usser Id as there will be a high probability that these person are going to be buy more

## picking a random sample
final=df.sample(n=2000)

final=df[0:2000]
#checking for missing values
final.isna().sum()


# #### as data is so huge,so if your system takes a lot for the execution , u can considered some sample of data from entire data,
#     as may be some of you have not that much good specifications in terms of processor ,RAM & HArd Disk..
#     so according to system specifications,u can considered some sample of data,if u have not issue with your specifications,
#     u can go ahead with this bulky data


## picking a random sample
final=df.sample(n=2000)

#### check missing values in dataset
final.isna().sum()

#### Removing the Duplicates if any
final.duplicated().sum()



### Analyse Length of Comments whether Customers are going to give Lengthy comments or short one
# we want fo ding the lenght of each comment
len(final['Text'][0].split(' '))

# Automate the process

def calc_len(text):
    return (len(text.split(' ')))


final['Text_length']=final['Text'].apply(calc_len)


import plotly.express as px
px.box(final, y="Text_length")


#### Conclusion-->>
#  Seems to have Almost 50 percent users are going to give their Feedback limited to 50 words whereas there are only few users who are going give Lengthy Feedbacks

## Analyze score 
sns.countplot(final['Score'], palette="plasma")

### Text Pre-Processsing
final['Text'] =final['Text'].str.lower() # converts all the text to lower case
final.head(10)

final['Text'][164]

## it converts any number that finds and puts a space, WE CANNOT USE THIS LOGIC HERE
import re
re.sub('[^a-zA-Z]',' ',final['Text'][164])

#### drawback of this re.sub in this use-case is, it will remove some numerical data too & may be that numerical values matters alot
#### thats way, I am going to create my own logic over here,that will remove all the special character

#### logic to remove punctuations or all the special characters
# define punctuation (simeio stijhs)
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~''' #define all the punctuations

data= final['Text'][164]

# remove punctuation from the string
no_punct = "" #define what you want to do
for char in data: # does not work for numbers only for symbols
    if char not in punctuations:
        no_punct = no_punct + char

# display the unpunctuated string
no_punct

#### Create function to remove punctuations in your review
def remove_punc(review):
    import string
    punctuations =string.punctuation
    # remove punctuation from the string
    no_punct = ""
    for char in review:
        if char not in punctuations:
            no_punct = no_punct + char
    return no_punct

# apply the above function
final['Text'] =final['Text'].apply(remove_punc)

# 
final['Text'][164]


#### Removal of Stopwords
# like as, was

import nltk
from nltk.corpus import stopwords

review='seriously this product was as tasteless as they come there are much better tasting products out there but at 100 calories its better than a special k bar or cookie snack pack you just have to season it or combine it with something else to share the flavor'

# you have no stopwords only the ones that cound for you
re=[word for word in review.split(' ') if word not in set(stopwords.words('english'))]
str=''
for wd in re:
    str=str+wd
    str=str+' '
str

#### using join to convert list into string
re=[word for word in review.split(' ') if word not in set(stopwords.words('english'))]
' '.join(re)

#### perform this task using function as I have to apply this logic on my entire column
def remove_stopwords(review):
    return ' '.join([word for word in review.split(' ') if word not in set(stopwords.words('english'))])

remove_stopwords(review)
final['Text'] = final['Text'].apply(remove_stopwords) # apply the function

### Pre-process your Data in a Depth
#### check if urls is present in Text column or not
final['Text'].str.contains('http?').sum()
final['Text'].str.contains('http?') # without the .sum it will show me exactly where these http are
final['Text'].str.contains('http').sum()

pd.set_option('display.max_rows',2000) # it will show all the values of the list
final['Text'].str.contains('http',regex=True)

#### we will observe we have some kind of URLs over here in my data that is definitely a kind of Dirtines in data, so we have to clean this data & make ready data for the analysis purpose

####  Removal of urls
import re
url_pattern = re.compile(r'href|http.\w+') # whether it has href or http , the . means that i have more words after the http
url_pattern.sub(r'', review)


# Automate the process
import re
def remove_urls(review):
    url_pattern = re.compile(r'href|http.\w+')
    return url_pattern.sub(r'', review)

final['Text'] = final['Text'].apply(remove_urls) ## apply the function
final['Text'] = final['Text'].apply(remove_urls)

for i in range(len(final['Text'])):
    final['Text'][i]=final['Text'][i].replace('br','')

import re
def remove_urls(review):
    url_pattern = re.compile(r'href|http.\w+')
    return url_pattern.sub(r'', review)



data2=final.copy()
stopwords = set(STOPWORDS) 

comment_words = '' 
for val in data2['Text']:
    # typecaste each val to string
    
    # split the value 
    tokens = val.split() 
    
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
    comment_words=comment_words+ " ".join(tokens)+" "

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 

# plot the WordCloud image                        
plt.figure(figsize = (8, 8)) 
plt.imshow(wordcloud) 
plt.axis("off") 