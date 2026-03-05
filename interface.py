
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import joblib

# ---- تحميل النماذج الحقيقية ----
nb_model = joblib.load('nb_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
bn_model = joblib.load('bn_model.pkl')

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
root.title("Email Spam Predictor")

tk.Label(root, text="اكتب البريد هنا:").pack()
email_text = tk.Text(root, height=10, width=60)
email_text.pack()

tk.Button(root, text="توقع", command=predict_email).pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 12))
result_label.pack(pady=10)

root.mainloop()
