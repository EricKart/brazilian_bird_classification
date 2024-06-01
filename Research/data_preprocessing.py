import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle

# Load the dataset
df = pd.read_csv("data/brazilian_migratory_birds.csv")

# Processing migratory status
df["is_migratory"] = df["migrant"].apply(lambda x: 1 if x == "MGT" else 0)

# Assuming using 'description' for now; adjust as needed
X = df["description"].astype("U")  # Ensure all text is treated as Unicode
y = df["is_migratory"]

# Vectorize text data
vectorizer = TfidfVectorizer(stop_words="english")
X_vectorized = vectorizer.fit_transform(X)

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# Save the processed data and vectorizer for later use
with open("processed_data.pkl", "wb") as f:
    pickle.dump((X_train, X_test, y_train, y_test), f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
