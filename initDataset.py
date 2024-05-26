import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as tf_hub
import tensorflow_text as tf_text
from sklearn.model_selection import train_test_split

def init_dataset():
    dataset_directory="Dataset/IMDB_Dataset.csv"
    df_dataset=pd.read_csv(dataset_directory)
    missing_values = df_dataset[df_dataset.isnull().any(axis=1)] # there wasn't missing column
    duplicates = df_dataset[df_dataset.duplicated(subset=["review", "sentiment"], keep=False)]

    #I dropped duplicated rows from dataset
    cleaned_df = df_dataset.drop_duplicates(subset=["review", "sentiment"])
    df = cleaned_df.dropna()
    df = df.sample(frac=0.05, random_state=42)

    print(df.sentiment.value_counts(normalize = True))

    # convert categorical target to numeric
    def convert_target(value):
        if value=="positive":
            return 1
        else:
            return 0
        
    df['sentiment']  =  df['sentiment'].apply(convert_target)
    print(df.value_counts)
    X_train, X_test, Y_train, Y_test=train_test_split(df["review"],df["sentiment"],stratify=df["sentiment"])
    return X_train, X_test, Y_train, Y_test

