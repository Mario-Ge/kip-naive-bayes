import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =====================================
# LOAD MODEL
# =====================================

print("=" * 60)
print("TESTING & EVALUASI MODEL")
print("=" * 60)

print("\n[1] Load model...")

model = joblib.load(
    "model/model.pkl"
)

prep = joblib.load(
    "model/preprocessor.pkl"
)

print("Model berhasil di-load!")

# =====================================
# LOAD DATA TESTING
# =====================================

print("\n[2] Load data testing...")

df_test = pd.read_csv(
    "data/data-testing-40-data-ok.csv"
)

print("Data testing berhasil di-load!")

# =====================================
# VALIDASI DATA
# =====================================

print("\n[3] Validasi dataset...")

prep.validate_data(df_test)

print("Dataset valid!")

# =====================================
# PISAHKAN FITUR & TARGET
# =====================================

print("\n[4] Pisahkan fitur dan target...")

metadata, X_test, y_test = (
    prep.separate_features_target(df_test)
)

# =====================================
# ENCODE DATA TESTING
# =====================================

print("\n[5] Encode data testing...")

X_test_encoded = prep.transform_new_data(
    X_test
)

y_test_encoded = prep.encoders[
    "target"
].transform(y_test)

print("Encoding selesai!")

# =====================================
# PREDIKSI
# =====================================

print("\n[6] Prediksi model...")

y_pred = model.predict(
    X_test_encoded
)

print("Prediksi selesai!")

# =====================================
# DECODE HASIL PREDIKSI
# =====================================

y_pred_label = prep.encoders[
    "target"
].inverse_transform(y_pred)

# =====================================
# TAMPILKAN HASIL PREDIKSI
# =====================================

print("\n[7] Hasil Prediksi")

hasil_df = pd.DataFrame({

    "kode_pendaftar":
    metadata["kode_pendaftar"],

    "nama":
    metadata["nama"],

    "actual":
    y_test.values,

    "predicted":
    y_pred_label
})

print(hasil_df.head(10))

# =====================================
# AKURASI
# =====================================

print("\n[8] Accuracy")

accuracy = accuracy_score(
    y_test_encoded,
    y_pred
)

print(f"Akurasi: {accuracy:.4f}")

# =====================================
# CONFUSION MATRIX
# =====================================

print("\n[9] Confusion Matrix")

cm = confusion_matrix(
    y_test_encoded,
    y_pred
)

print(cm)

# =====================================
# CLASSIFICATION REPORT
# =====================================

print("\n[10] Classification Report")

target_names = prep.encoders[
    "target"
].classes_

report = classification_report(
    y_test_encoded,
    y_pred,
    target_names=target_names
)

print(report)

print("=" * 60)
print("TESTING SELESAI")
print("=" * 60)