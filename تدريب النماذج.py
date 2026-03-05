# -------------------------
# train_compare_models.py
# -------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator

# -------------------------
# 1️⃣ إنشاء قاعدة بيانات صغيرة تجريبية (~700 إيميل)
# -------------------------
np.random.seed(42)
n_samples = 700
data = {
    "cleaned_body": ["This is email number {}".format(i) for i in range(n_samples)],
    "label": np.random.choice([0, 1], n_samples),  # 0=not spam, 1=spam
    "sender": np.random.choice(["alice@example.com", "bob@example.com", "carol@example.com"], n_samples),
    "subject": np.random.choice(["Hello", "Urgent", "Invoice", "Reminder"], n_samples),
    "hour": np.random.randint(0, 24, n_samples),
    "day": np.random.randint(1, 31, n_samples),

    "attachment": np.random.choice([0,1], n_samples),
    "priority": np.random.choice(["low","normal","high"], n_samples)
}
df = pd.DataFrame(data)

# -------------------------
# 2️⃣ تجهيز TF-IDF للـ Naive Bayes
# -------------------------
vectorizer = TfidfVectorizer(max_features=500)
X_nb = vectorizer.fit_transform(df['cleaned_body'])
y_nb = df['label']

# تقسيم البيانات
X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(X_nb, y_nb, test_size=0.2, random_state=42)


# تدريب Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_nb, y_train_nb)

# توقع
y_pred_nb = nb_model.predict(X_test_nb)

# -------------------------
# 3️⃣ تجهيز الخصائص للBayesian Network
# -------------------------
def extract_bn_features(row):
    text_lower = row['cleaned_body'].lower()
    return {
        "has_link": int("http" in text_lower or "www" in text_lower),
        "has_urgent": int("urgent" in text_lower),
        "has_password": int("password" in text_lower),
        "has_verify": int("verify" in text_lower),

        "num_urls": int(text_lower.count("http")),
        "body_length": len(text_lower),
        "attachment": row['attachment'],
        "priority": 1 if row['priority']=="high" else 0,
        "label": row['label']
    }

bn_features_list = [extract_bn_features(row) for _, row in df.iterrows()]
X_bn_df = pd.DataFrame(bn_features_list)

# تعريف الهيكل البنائي لـ BN (كل ميزة تؤثر على label)
features = ['has_link','has_urgent','has_password','has_verify','num_urls','body_length','attachment','priority']
edges = [(f,'label') for f in features]

bn_model = BayesianModel(edges)

# تدريب BN
bn_model.fit(X_bn_df, estimator=MaximumLikelihoodEstimator)

# -------------------------
# 4️⃣ تقييم الأداء Naive Bayes
# -------------------------
print("=== Naive Bayes Metrics ===")
print("Accuracy:", accuracy_score(y_test_nb, y_pred_nb))
print("Precision:", precision_score(y_test_nb, y_pred_nb, zero_division=0))
print("Recall:", recall_score(y_test_nb, y_pred_nb, zero_division=0))
print("F1 Score:", f1_score(y_test_nb, y_pred_nb, zero_division=0))

cm_nb = confusion_matrix(y_test_nb,

y_pred_nb)
ConfusionMatrixDisplay(cm_nb).plot()
plt.title("Naive Bayes Confusion Matrix")
plt.show()

# -------------------------
# 5️⃣ تقييم الأداء Bayesian Network
# -------------------------
X_train_bn, X_test_bn = train_test_split(X_bn_df, test_size=0.2, random_state=42)
y_test_bn = X_test_bn['label'].values
X_test_bn_features = X_test_bn.drop('label', axis=1)

y_pred_bn = bn_model.predict(X_test_bn_features)['label'].values

print("=== Bayesian Network Metrics ===")
print("Accuracy:",

accuracy_score(y_test_bn, y_pred_bn))
print("Precision:", precision_score(y_test_bn, y_pred_bn, zero_division=0))
print("Recall:", recall_score(y_test_bn, y_pred_bn, zero_division=0))
print("F1 Score:", f1_score(y_test_bn, y_pred_bn, zero_division=0))

cm_bn = confusion_matrix(y_test_bn, y_pred_bn)
ConfusionMatrixDisplay(cm_bn).plot()
plt.title("Bayesian Network Confusion Matrix")
plt.show()

# -------------------------
# 6️⃣ حفظ النماذج
# -------------------------
joblib.dump(nb_model, "nb_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

joblib.dump(bn_model, "bn_model.pkl")

print("✅ التدريب اكتمل والنماذج حفظت على القرص.")
