import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
import plotly.express as px 

# Load the trained model and test data
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('processed_data.pkl', 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

# Model Prediction and Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix of Model Predictions')
plt.show()

# Load the original dataset for comprehensive visualization
df = pd.read_csv('data/brazilian_migratory_birds.csv')
df['migratory_status'] = df['migrant'].apply(lambda x: 'Migratory' if x == 'MGT' else 'Resident')

# Visualization of Migratory vs. Resident Birds Status
summary_df = df['migratory_status'].value_counts().reset_index()
summary_df.columns = ['Migratory Status', 'Count']
plt.figure(figsize=(8, 6))
plt.bar(summary_df['Migratory Status'], summary_df['Count'], color=['skyblue', 'lightgreen'])
plt.xlabel('Migratory Status')
plt.ylabel('Count')
plt.title('Count of Migratory vs. Resident Birds in Dataset')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Enhanced Visualization with Families
df['BirdID'] = np.arange(len(df))  # Assign a numeric ID for visualization
top_families = df['familia'].value_counts().nlargest(10).index  # Focus on top families
df_filtered = df[df['familia'].isin(top_families)]

# Static Plot for Top Bird Families
plt.figure(figsize=(14, 10))
sns.scatterplot(data=df_filtered, x='BirdID', y='familia', hue='migratory_status', style='familia', s=100, palette={'Migratory':'blue', 'Resident':'red'})
plt.title('Individual Bird Migratory Status by Family (Top Families)')
plt.xlabel('Bird ID')
plt.ylabel('Family')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Status / Family')
plt.tight_layout()
plt.show()

# Interactive Plot for Individual Bird Migratory Status by Family
fig = px.scatter(df_filtered, x='BirdID', y='familia', color='migratory_status', symbol='familia',
                 hover_data=['latin name', 'english'], title='Interactive View: Migratory Status by Family')
fig.update_traces(marker=dict(size=10))
fig.update_layout(legend_title_text='Status / Family')
fig.update_yaxes(categoryorder='total ascending')
fig.show()
