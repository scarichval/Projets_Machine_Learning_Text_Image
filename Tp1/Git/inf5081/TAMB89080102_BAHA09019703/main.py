from preprocessing import DataCleaner 
from comparaison_full_features import ClassifierAnalysis
from selection_information_gain import FeatureSelector
import pandas as pd

def main():
    # Créez une instance de DataCleaner
    cleaner = DataCleaner()
    
    file_paths = {
        'legitimate_users': 'Datasets/legitimate_users.txt',
        'legitimate_followings': 'Datasets/legitimate_users_followings.txt',
        'content_polluters': 'Datasets/content_polluters.txt', 
        'polluters_followings': 'Datasets/content_polluters_followings.txt',
        'legitimate_users_tweets': 'Datasets/legitimate_users_tweets.txt',
        'content_polluters_tweets': 'Datasets/content_polluters_tweets.txt'
    }

    dfs_profiles = [] 
    # Traitement des fichiers de profils d'utilisateurs
    for key in ['legitimate_users', 'content_polluters']:
        df_profile = cleaner.load_data(file_paths[key])
        df_profile = cleaner.rename_profile_columns(df_profile)
        
        # Ajouter la colonne 'Label' en fonction du type d'utilisateur
        if key == 'legitimate_users':
            df_profile['Label'] = 0
        else:
            df_profile['Label'] = 1
        
        columns_to_scale = ['NumberOfFollowings', 'NumberOfFollowers', 'NumberOfTweets'] 
        df_profile = cleaner.prepare_dataset(df_profile, columns_to_scale)
        df_profile = cleaner.add_profile_features(df_profile)

        # Stocker chaque DataFrame traité dans la liste
        dfs_profiles.append(df_profile)
        
    df_final_profiles = pd.concat(dfs_profiles, ignore_index=True)

    dfs_followings = []
    # Traitement des fichiers de followings
    for key in ['legitimate_followings', 'polluters_followings']:
        df_followings = cleaner.load_data(file_paths[key])
        df_followings = cleaner.prepare_followings_dataset(df_followings)
        df_followings = cleaner.add_followings_features(df_followings)

        # Stocker chaque DataFrame traité dans la liste
        dfs_followings.append(df_followings)

        df_final_followings = pd.concat(dfs_followings, ignore_index=True)
        
    # fusion des caracteristiques de profil et de following
    df_final = cleaner.merge_features(df_final_profiles, df_final_followings)
    
    dfs_tweets = []
    for key in ['legitimate_users_tweets', 'content_polluters_tweets']:
        df_tweets = cleaner.load_data(file_paths[key])
        df_tweets = cleaner.rename_tweets_columns(df_tweets)
        df_tweets = cleaner.prepare_tweets_dataset(df_tweets)

        # Stocker chaque DataFrame traité dans la liste
        dfs_tweets.append(df_tweets)

        df_final_tweets = pd.concat(dfs_tweets, ignore_index=True)
        
    # Fusionner df_tweets avec df_final
    df_final = df_final.merge(df_final_tweets, on="UserID", how="left")  

    # df_final: le dataframe pour la tache 1 avec les 15 caracteristiques
    df_final = cleaner.clean_data(df_final)
    df_final = cleaner.reorganize_df(df_final) 

    df_final.to_csv('df_final_Merge.csv', index=False)  # Exporter en CSV

    # Comparaison des algorithmes pour la tache 1
    analysis = ClassifierAnalysis('df_final_Merge.csv')
    analysis.run_analysis()

    print("\n#############################")
    # Calcul du gain d'information pour la tache 2
    selector = FeatureSelector('df_final_Merge.csv')
    selector.prepare_data()
    selected_features_df = selector.select_features_with_info_gain(k=7)

    print("\n#############################")
    # pretraitement et construction du dataframe de 7 caracteristique pertinentes
    # df_final2: le dataframe pour la tache 2 avec les 7 caracteristique pertinentes
    df_final2 = df_final[selected_features_df.columns.tolist() + ['Label']]
    df_final2 = cleaner.clean_data(df_final2)

    df_final2.to_csv('df_final_GainInfo.csv', index=False)  # Exporter en CSV

    print("\n#############################")
    # Comparaison des algorithmes pour la tache 2
    analysis2 = ClassifierAnalysis('df_final_GainInfo.csv')
    analysis2.run_analysis()
    
if __name__ == "__main__":
    main()
