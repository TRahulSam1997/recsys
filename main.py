import numpy as np
import scipy
import pandas as pd
import math
import random
import sklearn
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

articles_df = pd.read_csv('./input/shared_articles.csv')
articles_df = articles_df[articles_df['eventType'] == ['CONTENT SHARED']]
articles_df.head(5)

print(articles_df)