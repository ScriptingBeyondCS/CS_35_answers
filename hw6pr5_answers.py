import random
import math

import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import json
import requests
import textblob

import nltk
from nltk.corpus import opinion_lexicon




# FOR ORGANIZING ORIGINAL JSON FILE 
#--------------------------------------------------
# f = open('reviews_Apps_for_Android_5.json', 'r')
# string_data = f.read()
# f.close()

# new_string_data = '['
# count = 0
# for x in string_data:
#     if count == 25000:
#         break
#     elif x == '}':
#         if count == 24999:
#             new_string_data += '}'
#         else:
#             new_string_data += '},'
#         count +=1
#     else: 
#         new_string_data += x
# new_string_data += ']'


# count1 = 0
# count2 = 0
# count3 = 0
# count4 = 0
# count5 = 0
# new_data = []
# data = json.loads(new_string_data)
# for line in data:
#     if line['overall'] == 1.0 and count1 <= 1500:
#         new_data += [line]
#         count1 += 1
#     elif line['overall'] == 2.0 and count2 <= 1500:
#         new_data += [line]
#         count2 += 1
#     elif line['overall'] == 3.0 and count3 <= 1500:
#         new_data += [line]
#         count3 += 1
#     elif line['overall'] == 4.0 and count4 <= 1500:
#         new_data += [line]
#         count4 += 1
#     elif line['overall'] == 5.0 and count5 <= 1500:
#         new_data += [line]
#         count5 += 1
#     else:
#         pass

# string_data = json.dumps( new_data, indent=2 )
# f = open('app_reviews.json', 'w')
# f.write(string_data)
# f.close()
#----------------------------------------------------


print('Assembling data...')

f = open( 'app_reviews.json', "r" )
string_data = f.read()
dict_data = json.loads(string_data)

# random.seed(0)
random.shuffle(dict_data)


# Read all of the opinion words in from the nltk corpus.
pos=list(opinion_lexicon.words('positive-words.txt'))
neg=list(opinion_lexicon.words('negative-words.txt'))

#Store them as a set (it'll make our feature extractor faster).
pos_set = set(pos)
neg_set = set(neg)

def review_features(review):
    """feature engineering for product reviews"""

    # Sentiment Analysis for text
    positive_count = 0
    negative_count = 0
    not_count = 0

    blob = textblob.TextBlob(review['reviewText'])
    sentiment = blob.sentiment
    length = len(blob.words)

    for sentence in blob.sentences:
        if 'n\'t like' in sentence or 'not like' in sentence:
            positive_count -= 1

    for word in blob.words:
        if word in pos_set: 
            positive_count += 1
        if word in neg_set or word == 'freeze': 
            negative_count += 1
        if word == 'not' or 'n\'t' in word:
            not_count += 1 
    
    # Sentiment Analysis for summary
    summary_positive_count = 0
    summary_negative_count = 0
    summary_not_count = 0

    blob = textblob.TextBlob(review['summary'])
    sentiment = blob.sentiment
    length = len(blob.words)

    for sentence in blob.sentences:
        if 'n\'t like' in sentence or 'not like' in sentence:
            summary_positive_count -= 1

    for word in blob.words:
        if word in pos_set: 
            summary_positive_count += 1
        if word in neg_set or word == 'freeze': 
            summary_negative_count += 1
        if word == 'not' or 'n\'t' in word:
            summary_not_count += 1 

    # Helpfulness
    total_help = review['helpful'][0] + review['helpful'][1]
    helpfulness = review['helpful'][0]
    
    # Product and year
    product_number = review['asin']
    year = review['reviewTime'][-4:]

    features = {'not summary': summary_not_count, 'pos summary': summary_positive_count, 'neg summary': summary_negative_count, 'year': year, 'product': product_number, 'helpfulness': helpfulness, 'help total': total_help, 'not': not_count, 'length': length, 'positive': positive_count, 'negative': negative_count, 'polarity': sentiment[0], 'subjectivity':sentiment[1]}
    return features

# Convert dictionary of features to an array (these are our inputs)
features = [review_features(review) for review in dict_data]
v = DictVectorizer(sparse=False)
X = v.fit_transform(features)

# For printing purposes, data is a list of tuples with (review content, input features, score)
data = []
X_data = []
y_data = []
for i in range(len(dict_data)):
    data += [(dict_data[i], X[i], dict_data[i]['overall'])]
    X_data += [data[i][1]]
    y_data += [data[i][2]]

# Assemble data into training, devtest, and test sets
X_train = X_data[:5000]
y_train = y_data[:5000]

X_devtest = X_data[5000:6500]
y_devtest = y_data[5000:6500]

X_test = X_data[6500:]
y_test = y_data[6500:]


print('Running regression...')         

# CROSS VALIDATION
#----------------------------------------------------
# max_depth = 40
# n_estimators = 100
# best_test_score = 0
# best_train_score = 0
# best_depth = 0
# best_n = 0

# for x in range(1,max_depth+2,5):
#     print('...')
#     for n in range(1,n_estimators+2, 10):
#         rforest = RandomForestRegressor(max_depth=x, n_estimators=n)    
#         train_score = 0
#         test_score = 0
#         # adapt for cross-validation (at least 10 runs w/ average test-score)
#         for i in range(5):
#             #
#             # split into our cross-validation sets...
#             #
#             cv_data_train, cv_data_test, cv_target_train, cv_target_test = \
#                 train_test_split(X_train, y_train, test_size=0.2) # random_state=0 

#             # fit the model using the cross-validation data
#             #   typically cross-validation is used to get a sense of how well it works
#             #   and tune any parameters, such as the max_depth and n_estimators here...
#             rforest.fit(cv_data_train, cv_target_train) 

#             train_score += rforest.score(cv_data_train,cv_target_train)
#             test_score += rforest.score(cv_data_test,cv_target_test)

#     if test_score > best_test_score:
#         best_test_score = test_score
#         best_train_score = train_score
#         best_depth = x
#         best_n = n
    
# print("CV testing-data score:", best_test_score/10)
# print("Best depth: ", best_depth)
# print("Best n-estimators: ", best_n)

#----------------------------------------------------
rforest = RandomForestRegressor(n_estimators=40, max_depth=200)
rforest.fit(X_train, y_train)
predictions = rforest.predict(X_devtest)

for i in range(len(predictions[:100])):
    score = str('{0:.1f}'.format(predictions[i])) + '   |   '
    print('{0:10} {1}'.format(score, y_devtest[i]))


print("Score on validation set:", rforest.score(X_test, y_test))
#----------------------------------------------------

# Determine how many predictions were within 1 star
one_star_count = 0
half_star_count = 0
predictions = rforest.predict(X_test)
for i in range(len(predictions)):
    if abs(predictions[i] - y_test[i]) <= 1:
        one_star_count += 1
        if abs(predictions[i] - y_test[i]) <= 0.5:
            half_star_count += 1

print(one_star_count, "out of", len(predictions), "predictions within 1 star (", \
    '{0:.1f}'.format(100*one_star_count/len(predictions)),"% )")
print(half_star_count, "out of", len(predictions), "predictions within 0.5 star (", \
    '{0:.1f}'.format(100*half_star_count/len(predictions)),"% )")


# Find an entry in the printout that had a big error and 
# use findError to determine what features caused it 
def findError(n):
    review = data[n+5000][0]
    features = review_features(review)
    score = data[n+5000][2]
    print(review, '\n')
    print(features,'\n')
    print(score)
    
