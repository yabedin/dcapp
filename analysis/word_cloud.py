import pandas as pd 
import numpy as np
import csv 
import nltk 
import os 
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import io
import base64

def generate_wordcloud(csv_path, dr_label):
    df = pd.read_csv(csv_path)

    # Get doctor's words only
    is_doctor = df['SPEAKER'] == dr_label
    df_dr = df[is_doctor]
    
    #Lowercase words
    df_dr['WORD'] = [str(i).lower() for i in df_dr['WORD']] 

    #remove stopwords
    stop = stopwords.words('english')
    additions = ['%hesitation', 'jenny', 'yeah', 'think', 'well', 'thing', 'yes', 'things', 'see', 'got']
    stop = stop + additions

    df_dr['without_stopwords'] = df_dr['WORD'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    df_replace = df_dr['without_stopwords']

    # Generate word cloud
    wordcloud = WordCloud().generate(' '.join(df_replace))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    # plt.show()

    # Encode image as base64
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)

    cloud_url = base64.b64encode(img.getvalue()).decode()

    return cloud_url

    
# generate_wordcloud('eac13_dr_speech.csv', 0)