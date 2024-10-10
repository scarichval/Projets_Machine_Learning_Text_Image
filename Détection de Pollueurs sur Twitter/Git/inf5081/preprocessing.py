import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import re
from datetime import timedelta

class DataCleaner:
    def __init__(self):
        self.scaler = StandardScaler()

    def load_data(self, file_path):
        # Charge les données à partir d'un fichier CSV.
        return pd.read_csv(file_path, delimiter="\t")  

    def clean_data(self, df):
        # Créez une copie du DataFrame pour éviter de modifier une vue d'un autre DataFrame
        df_copy = df.copy()

        # Supprimez les doublons
        df_copy.drop_duplicates(inplace=True)
        
        # Remplacez les valeurs NaN par la médiane pour chaque colonne numérique
        for col in df_copy.select_dtypes(include=np.number).columns:
            median_value = df_copy[col].median()
            df_copy.loc[:, col] = df_copy.loc[:, col].fillna(median_value)
        
        return df_copy


    def normalize_data(self, df, columns_to_scale):
        # Normalise seulement les colonnes spécifiées en utilisant la normalisation z-score.
        df[columns_to_scale] = self.scaler.fit_transform(df[columns_to_scale])
        return df

    def prepare_dataset(self, df, columns_to_scale):
        # Prépare le dataset complet pour l'analyse.
        df = self.clean_data(df)
        df = self.normalize_data(df, columns_to_scale)
        return df
    
    def convert_series_to_list(self, df, column_name):
        # Convertit des chaînes de nombres séparées en listes de nombres.  
        if column_name in df.columns:
             df[column_name] = df[column_name].apply(lambda x: [int(i) for i in x.split(',') if i.isdigit()])
        else:
             print(f"La colonne {column_name} n'existe pas dans ce DataFrame.")
        return df

    def prepare_followings_dataset(self, df):
        # Prépare le dataset de followings pour l'analyse. 
        df.columns = ['UserID','SeriesOfNumberOfFollowings']   
        df = self.convert_series_to_list(df, 'SeriesOfNumberOfFollowings')
        return df
    
    def rename_profile_columns(self, df):
        df.columns = ['UserID', 'CreatedAt', 'CollectedAt', 
                      'NumberOfFollowings', 'NumberOfFollowers', 
                      'NumberOfTweets', 'LengthOfScreenName', 'LengthOfDescriptionInUserProfile']
        return df
    
    def rename_tweets_columns(self, df):
        df.columns = ['UserID', 'TweetID', 'Tweet', 'CreatedAt']
        return df
        
    def prepare_tweets_dataset(self, df):
        # Nettoyage initial des données
        df = self.clean_data(df)
        
        # Convertir 'CreatedAt' en datetime pour faciliter les calculs temporels
        df['CreatedAt'] = pd.to_datetime(df['CreatedAt'])
        
        # Trier par UserID et CreatedAt pour garantir l'ordre chronologique
        df = df.sort_values(['UserID', 'CreatedAt'])
        
        # Calcul du nombre total de tweets par utilisateur
        df_total_tweets = df.groupby('UserID').size().reset_index(name='total_tweets')
        
        # Calcul de la durée de vie du compte en jours
        df_account_lifetime = df.groupby('UserID')['CreatedAt'].agg(['min', 'max'])
        df_account_lifetime['account_lifetime_days'] = (df_account_lifetime['max'] - df_account_lifetime['min']).dt.days.replace(0, 1)  # Remplacer 0 par 1 pour éviter la division par zéro
        
        # Calcul du nombre de tweets par jour
        df_tweets_per_day = df_total_tweets.copy()
        df_tweets_per_day['account_lifetime_days'] = df_account_lifetime['account_lifetime_days'].values
        df_tweets_per_day['tweets_per_day'] = df_tweets_per_day['total_tweets'] / df_tweets_per_day['account_lifetime_days']
        
        # Extraction et calcul du nombre d'URLs et de mentions dans chaque tweet
        df['url_count'] = df['Tweet'].astype(str).apply(lambda x: len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', x)))
        df['mention_count'] = df['Tweet'].astype(str).apply(lambda x: len(re.findall(r'@\w+', x)))
        
        # Calcul des moyennes des URLs et mentions par tweet pour chaque utilisateur
        df_url_mention_avg = df.groupby('UserID')[['url_count', 'mention_count']].mean().reset_index().rename(columns={'url_count': 'avg_urls_per_tweet', 'mention_count': 'avg_mentions_per_tweet'})
        
        # Calcul des temps entre les tweets
        df['time_diff'] = df.groupby('UserID')['CreatedAt'].diff().dt.total_seconds().div(60)  # Conversion en minutes
        df_time_diff_stats = df.groupby('UserID')['time_diff'].agg(['mean', 'max']).reset_index().rename(columns={'mean': 'mean_time_diff_min', 'max': 'max_time_diff_min'})
        
        # Calcul du rapport nombre de tweets par rapport à la durée de vie du compte
        df_tweets_lifetime_ratio = df_total_tweets.copy()
        df_tweets_lifetime_ratio['account_lifetime_days'] = df_account_lifetime['account_lifetime_days'].values
        df_tweets_lifetime_ratio['tweets_lifetime_ratio'] = df_tweets_lifetime_ratio['total_tweets'] / df_tweets_lifetime_ratio['account_lifetime_days']
        
        # Fusionner toutes les caractéristiques calculées dans un seul DataFrame
        df_features = df_total_tweets.merge(df_tweets_per_day[['UserID', 'tweets_per_day']], on='UserID')
        df_features = df_features.merge(df_url_mention_avg, on='UserID')
        df_features = df_features.merge(df_time_diff_stats, on='UserID')
        df_features = df_features.merge(df_tweets_lifetime_ratio[['UserID', 'tweets_lifetime_ratio']], on='UserID')
        
        return df_features

    def add_profile_features(self, df_profile):
        # Conversion des colonnes de date en datetime
        df_profile['CreatedAt'] = pd.to_datetime(df_profile['CreatedAt'])
        df_profile['CollectedAt'] = pd.to_datetime(df_profile['CollectedAt'])
        
        # Calcul de la durée de vie du compte en jours
        df_profile['account_lifetime_days'] = (df_profile['CollectedAt'] - df_profile['CreatedAt']).dt.days
        
        # Assurez-vous que les colonnes NumberOfFollowings et NumberOfFollowers sont numériques
        df_profile['NumberOfFollowings'] = pd.to_numeric(df_profile['NumberOfFollowings'], errors='coerce')
        df_profile['NumberOfFollowers'] = pd.to_numeric(df_profile['NumberOfFollowers'], errors='coerce')
        
        # Calcul du rapport following / followers pour chaque utilisateur
        # Éviter la division par zéro en remplaçant les zéros dans NumberOfFollowers par 1 pour ce calcul
        df_profile['following_followers_ratio'] = df_profile['NumberOfFollowings'] / df_profile['NumberOfFollowers'].replace(0, 1)
        
        return df_profile


    def add_followings_features(self, df_followings):
        # Calcul de l'écart type des IDs numériques des followings
        df_followings['std_followings'] = df_followings['SeriesOfNumberOfFollowings'].apply(lambda x: np.std(x) if x else 0)
        # Créer un nouveau DataFrame contenant uniquement UserID et les caractéristiques calculées
        df_followings_features = df_followings[['UserID', 'std_followings']]
        return df_followings_features
    
    def merge_features(self, df_profile_features, df_followings_features):
        # Fusionner les DataFrame sur UserID
        df_merged = pd.merge(df_profile_features, df_followings_features, on="UserID", how="left")
        return df_merged

    def reorganize_df(self, df_to_reorganize):
        columns_order = ['UserID', 'CreatedAt', 'CollectedAt', 'NumberOfFollowings', 'NumberOfFollowers',
                 'NumberOfTweets', 'LengthOfScreenName', 'LengthOfDescriptionInUserProfile',
                 'account_lifetime_days', 'following_followers_ratio', 'std_followings',
                 'total_tweets', 'tweets_per_day', 'avg_urls_per_tweet', 'avg_mentions_per_tweet',
                 'mean_time_diff_min', 'max_time_diff_min', 'tweets_lifetime_ratio', 'Label']
        
        # Réorganise le DataFrame selon l'ordre filtré
        df_to_reorganize = df_to_reorganize[columns_order]
        
        return df_to_reorganize
    
    def reorganize_df2(self, df_to_reorganize, columns_order):
        # Assurez-vous que toutes les colonnes dans columns_order existent dans df_to_reorganize
        columns_order = [col for col in columns_order if col in df_to_reorganize.columns]

        df_to_reorganize = df_to_reorganize[columns_order]
        
        return df_to_reorganize


