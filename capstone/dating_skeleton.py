import pandas as pd
import numpy as np
import re
import seaborn as sns
import string
import nltk
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

def text_process(text):
    '''
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Return the cleaned text as a list of words
    '''
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


#Create your df here:
df = pd.read_csv("profiles.csv")
print(df.job.head())
print(df.sex.head())
print(df.job.value_counts())
plt.hist(df.age, bins=20)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.xlim(16, 80)
plt.show()

df['gender'] = df.sex.map({'m':0,'f':1})
df.groupby(['gender']).size().plot.bar()

# =============================================================================
# totalMaleWordCount = 0
# totalFemaleWordCount = 0
# totalMaleCount = df.sex.str.count('m').sum()
# totalFemaleCount = df.sex.str.count('f').sum()
# totalPersonCounter = min(totalMaleCount,totalFemaleCount)
# print(totalPersonCounter)
# equalNumMaleWordCount = 0
# equalNumFemaleWordCount = 0
# iterationCounter = 0
# avgMaleWordCount = 0
# avgFemaleWordCount = 0
# maleCounter = 0
# femaleCounter = 0
# equalMaleCounter = 0
# # We need to look through each record and get word count and sum for each gender
# for row in df.itertuples():
#     currentAllEssayWordCount = 0
#     currentAllEssayWordCount += len(re.findall(r'\b\w+\b', re.sub("[^a-zA-Z]"," ",str(row.essay0))))
#     #print(currentAllEssayWordCount)
#     currentAllEssayWordCount += len(re.findall(r'\b\w+\b', re.sub("[^a-zA-Z]"," ",str(row.essay1))))
#     currentAllEssayWordCount += len(re.findall(r'\b\w+\b', re.sub("[^a-zA-Z]"," ",str(row.essay2))))
#     currentAllEssayWordCount += len(re.findall(r'\b\w+\b', re.sub("[^a-zA-Z]"," ",str(row.essay3))))
#     currentAllEssayWordCount += len(re.findall(r'\b\w+\b', re.sub("[^a-zA-Z]"," ",str(row.essay4))))
#     currentAllEssayWordCount += len(re.findall(r'\b\w+\b', re.sub("[^a-zA-Z]"," ",str(row.essay5))))
#     currentAllEssayWordCount += len(re.findall(r'\b\w+\b', re.sub("[^a-zA-Z]"," ",str(row.essay6))))
#     currentAllEssayWordCount += len(re.findall(r'\b\w+\b', re.sub("[^a-zA-Z]"," ",str(row.essay7))))
#     currentAllEssayWordCount += len(re.findall(r'\b\w+\b', re.sub("[^a-zA-Z]"," ",str(row.essay8))))
#     currentAllEssayWordCount += len(re.findall(r'\b\w+\b', re.sub("[^a-zA-Z]"," ",str(row.essay9))))
#     df.loc[row.Index, 'AllEssayWordCount'] = currentAllEssayWordCount
#     if row.sex == "m":
#         #print("Male")
#         totalMaleWordCount += currentAllEssayWordCount
#         maleCounter += 1
#         if iterationCounter <= totalPersonCounter:
#             iterationCounter += 1
#             equalNumMaleWordCount += currentAllEssayWordCount
#             equalMaleCounter += 1
#     if row.sex == "f":
#         #print("Female")
#         totalFemaleWordCount += currentAllEssayWordCount
#         femaleCounter += 1
# 
# 
# avgMaleWordCount = totalMaleWordCount / totalMaleCount
# avgFemaleWordCount = totalFemaleWordCount / totalFemaleCount
# print("Total number of males: {0}".format(str(totalMaleCount)))
# print("Total number of words from males: {0}".format(str(totalMaleWordCount)))
# print("Male counter: {0}".format(str(maleCounter)))
# print("Average words per total number of males: {0}".format(str(avgMaleWordCount)))
# print("Total number of females: {0}".format(str(totalFemaleCount)))
# print("Female counter: {0}".format(str(femaleCounter)))
# print("Total number of words from females: {0}".format(str(totalFemaleWordCount)))
# print("Average words per total number of females: {0}".format(str(avgFemaleWordCount)))
# avgMaleWordCount = equalNumMaleWordCount / totalPersonCounter
# print("Limit to minimum total number of people by gender: {0}".format(str(totalPersonCounter)))
# print("Equal gender ratio - finding male counter: {0}".format(str(equalMaleCounter)))
# print("Average words per male using equal ratio of men-to-women: {0}".format(str(avgMaleWordCount)))
# =============================================================================


#Questions:
#1. Can we predict age by drug use?
#2. Can we predict sex by essay? (Using Classification Techniques)
#3. Can we predict height by income level? (Using Regression Techniques)

#Requirements:
#1. 2 graphs containing exploration of the dataset
#2. a statement of your question (or questions!) and how you arrived there 
#3. explanation of at least two new columns you created and how you did it
#4. Comparison between two classification approaches
#    - including a qualitative discussion of simplicity
#    - time to run the model, and accuracy, precision, and/or recall
#5. Comparison between two regression approaches
#    - including a qualitative discussion of simplicity
#    - time to run the model, and accuracy, precision, and/or recall
#6. Overall conclusion, 
#    - with a preliminary answer to your initial question(s)
#    - next steps
#    - and what other data you would like to have in order to better answer your question(s)

#Augument data - transforming columns
#1.  Age Grouping   Length of Essay
#    18-23   0      
#    24-29   1
#    30-34   2
#    35-39   3
#    40-44   4
#    45-49   5
#    50-54   6
#    55-59   7
#    60-69   8
#    70 >    9

# Determine which column range contains essays
print(df.columns[6:16]) # Essay columns are 6-15 (range is always -1 of last element)
# =============================================================================
# target_names = np.array(['Male.Essays','Female.Essays'])
# targets = np.array([0,1])
# # add columns to your data frame
# df['is_train'] = np.random.uniform(0, 1, len(df)) <= 0.75
# df['Type'] = pd.Factor(targets, target_names)
# df['Targets'] = targets
# # define training and test sets
# train = df[df['is_train']==True]
# test = df[df['is_train']==False]
# trainTargets = np.array(train['Targets']).astype(int)
# testTargets = np.array(test['Targets']).astype(int)
# # columns you want to model
# features = df.columns[6:15]
# # call Gaussian Naive Bayesian class with default parameters
# gnb = GaussianNB()
# # train model
# y_gnb = gnb.fit(train[features], trainTargets).predict(train[features])
# =============================================================================

# Convert categorical variable to numeric
df["Gender"]=np.where(df["sex"]=="m",0,1)
# Clean essay columns - strings only
df.essay0.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
df.essay1.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
df.essay2.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
df.essay3.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
df.essay4.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
df.essay5.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
df.essay6.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
df.essay7.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
df.essay8.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
df.essay9.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
# =============================================================================
# df["Embarked_cleaned"]=np.where(data["Embarked"]=="S",0,
#                                   np.where(data["Embarked"]=="C",1,
#                                            np.where(data["Embarked"]=="Q",2,3)
#                                           )
#                                  )
# # Cleaning dataset of NaN
# =============================================================================
#df=df[["Sex_cleaned","essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]].dropna(axis=0, how='any')
#type(df)
essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]
# Removing the NaNs
all_essays = df[essay_cols].replace(np.nan, '', regex=True)
# Combining the essays
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)
df["Total Essay Length"] = all_essays.apply(lambda x: len(x))
print(df.head())

g = sns.FacetGrid(data=df, col='Gender')
g.set(xlim=(0, 10000))
g.map(plt.hist, 'Total Essay Length', bins=100)

gender_class = df[(df['Gender'] == 0) | (df['Gender'] == 1)]
print(gender_class.shape)

X = all_essays#gender_class[essay_cols]
y = gender_class['Gender']
#print(X[0])
#bow_transformer = CountVectorizer(analyzer=text_process).fit(X)
#len(bow_transformer.vocabulary_)