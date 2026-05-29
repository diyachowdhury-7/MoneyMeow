#Reading the CSV and categorizing it 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

df = pd.read_csv("transactions.csv")

#Converting to vector
vectorizer=TfidfVectorizer()
x=vectorizer.fit_transform(df["description"])   
y=df["category"]


#splitting data 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#the model
model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)

#checking accuracy
scores = cross_val_score(model, x, y, cv=5)
print("CV Scores:", scores)
print("Average Accuracy:", scores.mean())

#testing with new data using zip
new_trans=[
    "Zomato Biryani order",
    "Uber cab ride",
    "Amazon shopping",
    "IndiGo flight booking"
]
new_vec=vectorizer.transform(new_trans)
prediction=model.predict(new_vec)
for desc, pred in zip(new_trans, prediction):
    print(desc, "->", pred)

#graph
import matplotlib.pyplot as plt

df["predicted_category"] = model.predict(x)
category_totals = df.groupby("predicted_category")["amount"].sum()

category_totals.plot(kind="pie", autopct="%1.1f%%")
plt.title("Spending by Category")
plt.ylabel("")
plt.show()