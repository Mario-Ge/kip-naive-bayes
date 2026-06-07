import sys
import os

# =========================================
# TAMBAHKAN ROOT PROJECT
# =========================================

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

# =========================================
# IMPORT LIBRARY
# =========================================

import time
import tracemalloc
import psutil
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from core.preprocessing import Preprocessor
from core.naive_bayes import NaiveBayes

# =========================================
# HEADER
# =========================================

print("=" * 70)
print("BENCHMARK ALGORITMA NAIVE BAYES")
print("=" * 70)

# =========================================
# LOAD DATASET
# =========================================

print("\n[1] LOAD DATASET")

prep = Preprocessor()

# training dataset
_, X_train, y_train = prep.fit_transform(
    "data/Data-Trainning-160-data-ok.csv"
)

# testing dataset
df_test = pd.read_csv(
    "data/data-testing-40-data-ok.csv"
)

_, X_test, y_test = (
    prep.separate_features_target(df_test)
)

# encode testing
X_test = prep.transform_new_data(
    X_test
)

y_test = prep.encoders[
    "target"
].transform(y_test)

print("Dataset berhasil di-load!")

print(f"Jumlah data training : {len(X_train)}")
print(f"Jumlah data testing  : {len(X_test)}")
print(f"Jumlah fitur         : {X_train.shape[1]}")

# =========================================
# INISIALISASI MODEL
# =========================================

print("\n[2] INISIALISASI MODEL")

model = NaiveBayes()

print("Model berhasil dibuat!")

# =========================================
# BENCHMARK TRAINING
# =========================================

print("\n[3] BENCHMARK TRAINING")

process = psutil.Process(
    os.getpid()
)

# memory sebelum training
memory_before_train = (
    process.memory_info().rss / 1024 / 1024
)

# mulai tracking memory python
tracemalloc.start()

# mulai timer
start_train = time.perf_counter()

# =========================================
# TRAINING MODEL
# =========================================

model.fit(X_train, y_train)

# =========================================
# SELESAI TRAINING
# =========================================

# stop timer
end_train = time.perf_counter()

# ambil memory tracking
current_train, peak_train = (
    tracemalloc.get_traced_memory()
)

# stop tracking
tracemalloc.stop()

# memory sesudah training
memory_after_train = (
    process.memory_info().rss / 1024 / 1024
)

# total waktu training
training_time = (
    end_train - start_train
)

# total penggunaan memory
training_memory = (
    memory_after_train -
    memory_before_train
)

print("Training selesai!")

# =========================================
# BENCHMARK PREDICTION
# =========================================

print("\n[4] BENCHMARK PREDICTION")

# memory sebelum prediction
memory_before_predict = (
    process.memory_info().rss / 1024 / 1024
)

# tracking memory python
tracemalloc.start()

# timer mulai
start_predict = time.perf_counter()

# =========================================
# PREDIKSI
# =========================================

y_pred = model.predict(X_test)

# =========================================
# SELESAI PREDIKSI
# =========================================

# timer selesai
end_predict = time.perf_counter()

# memory tracking
current_predict, peak_predict = (
    tracemalloc.get_traced_memory()
)

# stop tracking
tracemalloc.stop()

# memory sesudah prediction
memory_after_predict = (
    process.memory_info().rss / 1024 / 1024
)

# waktu prediction
prediction_time = (
    end_predict - start_predict
)

# penggunaan memory prediction
prediction_memory = (
    memory_after_predict -
    memory_before_predict
)

print("Prediksi selesai!")

# =========================================
# EVALUASI MODEL
# =========================================

print("\n[5] EVALUASI MODEL")

accuracy = accuracy_score(
    y_test,
    y_pred
)

cm = confusion_matrix(
    y_test,
    y_pred
)

target_names = prep.encoders[
    "target"
].classes_

report = classification_report(
    y_test,
    y_pred,
    target_names=target_names
)

# =========================================
# HASIL BENCHMARK
# =========================================

print("\n" + "=" * 70)
print("HASIL BENCHMARK")
print("=" * 70)

# =========================================
# TRAINING
# =========================================

print("\n[TRAINING PERFORMANCE]")

print(
    f"Waktu Training        : "
    f"{training_time:.6f} detik"
)

print(
    f"Memory Training       : "
    f"{training_memory:.6f} MB"
)

print(
    f"Peak Memory Training  : "
    f"{peak_train / 1024 / 1024:.6f} MB"
)

# =========================================
# PREDICTION
# =========================================

print("\n[PREDICTION PERFORMANCE]")

print(
    f"Waktu Prediksi        : "
    f"{prediction_time:.6f} detik"
)

print(
    f"Memory Prediksi       : "
    f"{prediction_memory:.6f} MB"
)

print(
    f"Peak Memory Prediksi  : "
    f"{peak_predict / 1024 / 1024:.6f} MB"
)

# =========================================
# AKURASI
# =========================================

print("\n[MODEL ACCURACY]")

print(
    f"Akurasi Model         : "
    f"{accuracy:.4f}"
)

# =========================================
# CONFUSION MATRIX
# =========================================

print("\n[CONFUSION MATRIX]")

print(cm)

# =========================================
# CLASSIFICATION REPORT
# =========================================

print("\n[CLASSIFICATION REPORT]")

print(report)

print("=" * 70)
print("BENCHMARK SELESAI")
print("=" * 70)