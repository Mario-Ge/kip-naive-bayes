import joblib

from core.preprocessing import Preprocessor
from core.naive_bayes import NaiveBayes


def main():

    print("=" * 60)
    print("TRAINING MODEL NAIVE BAYES")
    print("=" * 60)

    # =====================================
    # PREPROCESSING
    # =====================================

    print("\n[1] Load dan preprocessing dataset...")

    prep = Preprocessor()

    metadata, X_train, y_train = prep.fit_transform(
        "data/Data-Trainning-160-data-ok.csv"
    )

    print("Preprocessing selesai!")

    # =====================================
    # INFORMASI DATASET
    # =====================================

    print("\n[2] Informasi dataset")

    print(f"Jumlah data training : {len(X_train)}")
    print(f"Jumlah fitur         : {X_train.shape[1]}")

    print("\nNama fitur:")

    for col in X_train.columns:
        print(f"- {col}")

    # =====================================
    # TRAINING MODEL
    # =====================================

    print("\n[3] Training model Naive Bayes...")

    model = NaiveBayes()

    model.fit(X_train, y_train)

    print("Training selesai!")

    # =====================================
    # TAMPILKAN PRIOR
    # =====================================

    print("\n[4] Prior Probability")

    classes = model.get_classes()
    priors = model.get_prior()

    for i, c in enumerate(classes):

        label = prep.encoders["target"].inverse_transform(
            [c]
        )[0]

        print(
            f"{label} : {priors[i]}"
        )

    # =====================================
    # TAMPILKAN ENCODER
    # =====================================

    print("\n[5] Encoder Mapping")

    for column, encoder in prep.encoders.items():

        print(f"\n{column}")

        for index, label in enumerate(
            encoder.classes_
        ):
            print(f"{label} -> {index}")

    # =====================================
    # SIMPAN MODEL
    # =====================================

    print("\n[6] Menyimpan model...")

    joblib.dump(
        model,
        "model/model.pkl"
    )

    print("Model berhasil disimpan!")

    # =====================================
    # SIMPAN PREPROCESSOR
    # =====================================

    print("\n[7] Menyimpan encoder/preprocessor...")

    joblib.dump(
        prep,
        "model/preprocessor.pkl"
    )

    print("Preprocessor berhasil disimpan!")

    print("\n" + "=" * 60)
    print("TRAINING SELESAI")
    print("=" * 60)


if __name__ == "__main__":
    main()