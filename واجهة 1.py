
import tkinter as tk
from tkinter import messagebox
import pandas as pd

# ---- نماذج Dummy للتجربة ----
class DummyModel:
    def predict(self, X):
        # Naive Bayes أو Bayesian Network
        if isinstance(X, pd.DataFrame):
            # BN: ارجع عمود Label
            return pd.DataFrame({"Label": [1]}) if "urgent" in X['body_text'][0].lower() else pd.DataFrame({"Label":[0]})
        else:
            # Naive Bayes: ارجع تصنيف
            return ["Spam"] if "urgent" in X[0].lower() else ["not spam"]

class DummyVectorizer:
    def transform(self, texts):
        # ترجع نفس النصوص بدون تعديل (فقط لتوافق الكود)
        return texts

# ---- تحميل النماذج Dummy ----
nb_model = DummyModel()
vectorizer = DummyVectorizer()
bn_model = DummyModel()

# ---- دالة للتنبؤ ----
def predict_email():
    text = email_text.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("تنبيه", "اكتب نص البريد أولاً!")
        return

    # Naive Bayes prediction
    X_vec = vectorizer.transform([text])
    y_nb_pred = nb_model.predict(X_vec)[0]

    # Bayesian Network features
    features = {
        "body_text": text,
        "has_link": int("http" in text or "www" in text),

        "has_urgent": int("urgent" in text),
        "has_password": int("password" in text),
        "has_verify": int("verify" in text),
        "num_urls": int(text.count("http")),
        "body_length": len(text),
        "Label": 0
    }
    X_bn_test = pd.DataFrame([features])
    y_bn_pred = bn_model.predict(X_bn_test)['Label'].values[0]

    # عرض النتائج
    result_label.config(text=f"Naive Bayes: {y_nb_pred}\nBayesian Network: {y_bn_pred}")

# ---- إعداد نافذة ----
root = tk.Tk()
root.title("Email Spam Predictor (Demo)")

tk.Label(root, text="اكتب البريد هنا:").pack()
email_text = tk.Text(root, height=10, width=60)
email_text.pack()

tk.Button(root, text="توقع", command=predict_email).pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 12))
result_label.pack(pady=10)

root.mainloop()
