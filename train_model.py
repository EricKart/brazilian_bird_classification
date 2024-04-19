import pickle
from sklearn.linear_model import LogisticRegression
from scipy.sparse import issparse

# Load processed data
with open('processed_data.pkl', 'rb') as file:
    X_train, X_test, y_train, y_test = pickle.load(file)

# Ensure y_train is writable
y_train = y_train.copy()

# Logistic Regression Model
model = LogisticRegression(max_iter=1000)

# Check if X_train is a sparse matrix and ensure it's writable
if issparse(X_train):
    X_train.sort_indices()

model.fit(X_train, y_train)

# Save the trained model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
