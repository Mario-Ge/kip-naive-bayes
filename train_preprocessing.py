from core.preprocessing import Preprocessor


def main():

    print("=" * 50)
    print("TRAINING DATA PREPROCESSING")
    print("=" * 50)

    # inisialisasi preprocessing
    prep = Preprocessor()

    # load + preprocessing dataset
    metadata, X, y = prep.fit_transform(
        "data/Data-Trainning-160-data-ok.csv"
    )

    # tampilkan metadata
    print("\nMETADATA:")
    print(metadata.head())

    # tampilkan fitur hasil encoding
    print("\nFITUR HASIL ENCODING:")
    print(X.head())

    # tampilkan target hasil encoding
    print("\nTARGET HASIL ENCODING:")
    print(y[:10])

    # informasi dataset
    print("\nINFO DATASET:")
    print(f"Jumlah data: {len(X)}")
    print(f"Jumlah fitur: {X.shape[1]}")

    print("\nNAMA FITUR:")
    print(list(X.columns))

    # tampilkan encoder mapping
    print("\nENCODER MAPPING:")

    for column, encoder in prep.encoders.items():

        print(f"\n{column}")

        for index, label in enumerate(
            encoder.classes_
        ):
            print(f"{label} -> {index}")

    print("\nPreprocessing berhasil!")
    print("=" * 50)


if __name__ == "__main__":
    main()