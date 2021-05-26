import mysql.connector
import pandas as pd
import numpy as np
import contractions
import csv
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from textblob import TextBlob
nltk.download('wordnet')
nltk.download('stopwords')
from sklearn.cluster import KMeans
# from spacy import displacy
from sklearn.feature_extraction.text import TfidfVectorizer

def import_data():
   db_connect = mysql.connector.connect(host='127.0.0.1', database='mlmodel', user = 'root', password = '')
   df = pd.read_sql('SELECT * from mytable', con=db_connect)
   return df

class preprocessing():
   def __init__(self, df):
      self.df = df
   def drop_null_values(self,df):
      self.df = self.df.dropna()
      self.df.drop(self.df.columns[[0]], axis=1, inplace=True)
      return df

   def remove_contraction(self,df):
      df['title'] = df['title'].apply(lambda x: ' '.join([contractions.fix(word) for word in x.split()]))
      df['description'] = df['description'].apply(lambda x: ' '.join([contractions.fix(word) for word in x.split()]))
      return df
   def noise_cleaning(self,df):
# # Cleaning - spacing, special characters, lowercasing
      df['title'] = df['title'].apply(lambda x: ' '.join([x for x in x.split() if len(x) >= 3]))
      df['description'] = df['description'].apply(lambda x: ' '.join([x for x in x.split() if len(x) >= 3]))

      df['title'] = df['title'].apply(lambda x: x.lower())
      df['description'] = df['description'].apply(lambda x: x.lower())

      df['title'] = df['title'].str.replace('[^\w\s]',' ')
      df['description'] = df['description'].str.replace('[^\w\s]', ' ')
      df['title'] = df['title'].str.replace('\d+',' ')
      df['description'] = df['description'].str.replace('\d+', ' ')
      return df

   def tockenization(self,df):
      stop=stopwords.words('english')
      df['tokenized_title'] = df['title'].apply(word_tokenize)
      df['tokenized_description'] = df['description'].apply(word_tokenize)


      df['title'] = df['title'].apply(lambda x: ' '.join([x for x in x.split() if x not in stop]))
      df['description'] = df['description'].apply(lambda x: ' '.join([x for x in x.split() if x not in stop]))
      df['title_stopword_removed'] = df['tokenized_title'].apply(lambda x: [word for word in x if word not in (stop)])
      df['description_stopword_removed'] = df['tokenized_description'].apply(lambda x: [word for word in x if word not in (stop)])
      return df

   # def Lemmatization(self,df):

class Vecorization_and_SVD():
   def __init__(self, df):
      self.df = df
   def tfidf(self,df):
      # tfidf vectorizer of scikit learn
      vectorizer = TfidfVectorizer()
      X = vectorizer.fit_transform(self.df['description'])

      # print("Shape of matrix is",X.shape)  # check shape of the document-term matrix
      terms = vectorizer.get_feature_names()
      print(terms)
      return X

   def truncated_SVD(self,X):
      from sklearn.decomposition import TruncatedSVD
      svd = TruncatedSVD(100)
      transformed = svd.fit_transform(X)
      return transformed

class clustering():
   def __init__(self, df,transformed):
      self.df = df
      self.transformed=transformed

   def elbow_method(self,transformed):
      distortion = []
      for i in range(1, 200):
         kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
         kmeans.fit(self.transformed)
         distortion.append(kmeans.inertia_)

      plt.plot(range(1, 200), distortion)
      plt.title('The Elbow Method')
      plt.xlabel('Number of Clusters')
      plt.ylabel('Distortion')
      plt.show()
#
   def clustering(self,transformed,df):
      num_clusters = 200
      km = KMeans(n_clusters=num_clusters)
      km.fit(self.transformed)
      clusters = km.labels_.tolist()
      print(clusters)
# # Labelling data with clusters
#       self.df['cluster_label'] = np.array(clusters)
#       self.df.drop(self.df.columns[[6, 7, 8, 9, 10, 11, 12]], axis=1, inplace=True)
#       # df.head()
#       df.to_csv("clustered_data")
#       return clusters


def main():
   df=import_data()
   pre =preprocessing(df)
   remove_null_df=pre.drop_null_values(df)
   contracted_df=pre.remove_contraction(remove_null_df)
   noise_cleaned_df=pre.noise_cleaning(contracted_df)
   tockenized_df=pre.tockenization(noise_cleaned_df)


   vec =Vecorization_and_SVD(df)
   vectorized=vec.tfidf(tockenized_df)
   transformed=vec.truncated_SVD(vectorized)


   clus=clustering(df,transformed)
   elbow_method=clus.elbow_method(transformed)
   clusters=clus.clustering(df,transformed)
   print(clusters)



if __name__ == "__main__":
   main()

