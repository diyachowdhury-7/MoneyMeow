# MoneyMeow: Personal Expense Categorizer

A supervised machine learning web app that automatically categorizes personal transactions from a CSV file using NLP.
Built with TF-IDF vectorization and a Random Forest classifier, served via Flask and deployed on Render.

You upload a file of transactions with a description column: the model reads the text, extracts features, and predicts
a spending category (Food, Travel, Shopping, Transport, etc.) for each row.
Results are displayed as a categorized breakdown with a pie chart.

Unlike manually tagging every transaction, this learns from labeled examples and predicts categories automatically
without you touching each row.

---

## ML pipeline

- **Text preprocessing**: transaction descriptions are cleaned and lowercased
- **TF-IDF vectorization**: converts description text into numeric feature vectors; words frequent in one category but rare overall get higher weight
- **Classification**: Random Forest models trained on labeled transaction data; predictions made on new unseen descriptions
- **Output**: each transaction is assigned a predicted category, visualized as a pie chart

---

## Project structure

    cashflow/
    ├── app.py
    ├── model.py
    ├── templates/
    │   └── index.html
    ├── data/
    │   └── transactions.csv
    └── README.md

---

## Input format

Accepts `.csv`. Minimum required columns:

| date | description | amount |
|---|---|---|
| 2024-01-01 | Swiggy food delivery | 450 |
| 2024-01-02 | IRCTC train ticket | 1200 |

---

## What I learned

- ML project structure from data → model → deployment
- TF-IDF as a way to represent text numerically for classification
- Difference between generative (Naive Bayes) and ensemble (Random Forest) approaches
- Deploying a Flask ML app to a live URL via Render
