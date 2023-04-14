
from yelpapi import YelpAPI
import pandas as pd 
import requests
import urllib.parse
import json
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords 


api_key = 'Vamo2J5tHxwahz2CunAGT3Mqj_eax93XEdNW9orXuB5Y-a0Jr0Wrpof2GGcstnocGPZod0AlcOtCCnveUCyPBO_AAE6xdlXQeK--sSsyCYjYQwjkCWkCFZVSuZU1ZHYx'
yelp_api_key = YelpAPI(api_key)

#search_query 20 places
search_name = 'chicken wings'
search_location = 'Dallas, TX'
search_sort_by = 'rating' #best_match, rating, review_count, distance
search_limit = 20

search_results = yelp_api_key.search_query(term = search_name, location = search_location, sort_by = search_sort_by, limit = search_limit)
print(search_results)
print('\n')


#convert to pandas
wing_df = pd.DataFrame.from_dict(search_results['businesses'])
print(wing_df)
#wing_df.to_csv('wings_results')

#searching for review using alias 
reviews_alias = "wing-city-garland-4"
reviews = yelp_api_key.reviews_query(id = reviews_alias)
print(reviews) #list of reviews
print('\n')

for wing_reviews in reviews['reviews']:
    print(wing_reviews['text'])

results = pd.DataFrame.from_dict(reviews['reviews'])
print(results['text'])
#results.to_csv(f'{reviews_alias}_yelpapi_businesses_results.csv')
print('\n')


# getting tokens

for text in results['text']:
    token = nltk.word_tokenize(text)
    tags = nltk.pos_tag(token)
    print(tags)
    for tag in tags:
        if tag[1] == 'JJ' or tag[1] == 'JJS' or tag[1] == 'NN':
            print(tag[0])
print('\n')
#sentiment
analyer = SentimentIntensityAnalyzer()
for review in results['text']:
    sentiment_score = analyer.polarity_scores(review)
    print(review)
    print('\n')
    print(sentiment_score)
    