import pandas as pd
data = pd.read_csv("anime_dataset_pre.csv")
#data = data[["title", "popularity", "favorites", "studios"]]
#data.to_csv("anime_dataset_pre.csv", index=False)

print(data.head())
print(data.shape)
print(data.info())
print(data.describe())


#print(data.isnull().sum())
#data = data.dropna()
#data.to_csv("anime_dataset_pre.csv", index=False)

#print(data.duplicated().sum())
#data.drop_duplicates(inplace=True) 
#data.to_csv("anime_dataset_pre.csv", index=False) 

#from sklearn.preprocessing import LabelEncoder
#encoder = LabelEncoder()
#data["studios"] = encoder.fit_transform(data["studios"])
#data.to_csv("anime_dataset_pre.csv", index=False)

#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
#data[["popularity", "favorites"]] = scaler.fit_transform(data[["popularity", "favorites"]])
#data.to_csv("anime_dataset_pre.csv", index=False)