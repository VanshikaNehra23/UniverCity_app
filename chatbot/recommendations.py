import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from operator import itemgetter

data=pd.read_csv("UCoursera_Courses.csv")
data.drop(["Unnamed: 0"],axis=1,inplace=True)

def remove(myString):
  myString = re.sub(r"[\n\t]*", "", myString)
  myString = re.sub(r'[.,"\'-?:!;]', '', myString)
  return myString

nltk.download('stopwords')
stopwords=nltk.corpus.stopwords.words('english')
def clean_text(txt):
    txt="".join([c for c in txt if c not in string.punctuation])
    tokens=re.split('\W+',txt)
    txt=[word for word in tokens if word not in stopwords]
    return txt

nltk.download('wordnet')
wn=nltk.WordNetLemmatizer()
def lemm(token_txt):
    text=[wn.lemmatize(word) for word in token_txt]
    return text

data['course_title_l']= data['course_title'].apply(lambda x: remove(x))
data['course_organization_l']= data['course_organization'].apply(lambda x: remove(x))
data['course_title_l']=data['course_title_l'].apply(lambda x:clean_text(x))
data['course_organization_l']= data['course_organization_l'].apply(lambda x: clean_text(x))
data['course_title_l']=data['course_title_l'].apply(lambda x: lemm(x))
data['course_organization_l']=data['course_organization_l'].apply(lambda x: lemm(x))

data['course_Certificate_type']=data['course_Certificate_type'].mask(data['course_Certificate_type']=='PROFESSIONAL CERTIFICATE','PROFESSIONAL_CERTIFICATE')

def create_soup(x):
    return ' '.join(x['course_title_l']) + ' ' + ' '.join(x['course_organization_l']) + ' ' + x['course_Certificate_type'] + ' ' + x['course_difficulty']

data['soup'] = data.apply(create_soup, axis=1)

def get_course():
    course = input("What Course are you interested in (if multiple, please separate them with a comma)? [Type 'skip' to skip this question] ")
    courses = " ".join(["".join(n.split()) for n in course.lower().split(',')])
    return course

def get_searchTerms():
    searchTerms = "" 
    courses = get_course()
    if courses != 'skip':
        searchTerms = courses
    return searchTerms

def make_recommendation(data=data):
    searchTerms = get_searchTerms()
    searchTerms = remove(searchTerms)
    searchTerms = clean_text(searchTerms)
    searchTerms = lemm(searchTerms)
    new_row = data.iloc[-1,:].copy()
    new_row.iloc[-1] = " ".join(searchTerms)
    data = data.append(new_row)
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(data['soup'])  
    cosine_sim2 = cosine_similarity(count_matrix, count_matrix) #getting a similarity matrix
    sim_scores = list(enumerate(cosine_sim2[-1,:]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    ranked_titles = []
    for i in range(1, 11):
        indx = sim_scores[i][0]
        print(sim_scores[i][1])
        if (sim_scores[i][1] != 0):
            ranked_titles.append([data['course_title'].iloc[indx], data['course_rating'].iloc[indx], data['course_students_enrolled'].iloc[indx]])
#     ranked_titles= sorted(ranked_titles, key=itemgetter(1), reverse=True)
    if len(ranked_titles)==0:
        return -1
    else:
        return ranked_titles

make_recommendation()




