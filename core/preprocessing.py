import pandas as pd
from sklearn.preprocessing import LabelEncoder


class Preprocessor:

    def __init__(self):
        """
        Menyimpan semua encoder
        agar konsisten saat training dan prediksi
        """
        self.encoders = {}

    def load_data(self, file_path):
        """
        Load dataset dari file CSV
        """

        df = pd.read_csv(file_path)

        return df

    def validate_data(self, df):
        """
        Validasi dataset
        """

        # cek missing value
        if df.isnull().sum().sum() > 0:
            raise ValueError(
                "Dataset memiliki data kosong!"
            )

        # cek kolom wajib
        required_columns = [
            "kode_pendaftar",
            "nama",
            "penghasilan",
            "pekerjaan",
            "status",
            "tanggungan",
            "nilai_raport",
            "nilai_tes",
            "prestasi",
            "pkh",
            "kks",
            "kip",
            "target"
        ]

        for column in required_columns:
            if column not in df.columns:
                raise ValueError(
                    f"Kolom '{column}' tidak ditemukan!"
                )

        return True

    def separate_features_target(self, df):
        """
        Pisahkan metadata, fitur, dan target
        """

        # metadata
        metadata = df[[
            "kode_pendaftar",
            "nama"
        ]]

        # fitur
        X = df.drop(columns=[
            "kode_pendaftar",
            "nama",
            "target"
        ])

        # target
        y = df["target"]

        return metadata, X, y

    def encode_features(self, X):
        """
        Encode semua fitur kategorikal
        """

        X_encoded = X.copy()

        for column in X_encoded.columns:

            le = LabelEncoder()

            X_encoded[column] = le.fit_transform(
                X_encoded[column]
            )

            # simpan encoder
            self.encoders[column] = le

        return X_encoded

    def encode_target(self, y):
        """
        Encode label target
        """

        le = LabelEncoder()

        y_encoded = le.fit_transform(y)

        # simpan encoder target
        self.encoders["target"] = le

        return y_encoded

    def fit_transform(self, file_path):
        """
        Pipeline preprocessing lengkap
        """

        # load dataset
        df = self.load_data(file_path)

        # validasi dataset
        self.validate_data(df)

        # pisahkan fitur dan target
        metadata, X, y = self.separate_features_target(df)

        # encode fitur
        X_encoded = self.encode_features(X)

        # encode target
        y_encoded = self.encode_target(y)

        return (
            metadata,
            X_encoded,
            y_encoded
        )

    def transform_new_data(self, X_new):
        """
        Transform data baru menggunakan encoder lama
        Dipakai saat prediksi
        """

        X_encoded = X_new.copy()

        for column in X_encoded.columns:

            if column not in self.encoders:
                raise ValueError(
                    f"Encoder untuk kolom '{column}' tidak ditemukan!"
                )

            le = self.encoders[column]

            X_encoded[column] = le.transform(
                X_encoded[column]
            )

        return X_encoded

    def decode_target(self, y_pred):
        """
        Mengembalikan hasil prediksi angka
        menjadi label asli
        """

        le = self.encoders["target"]

        return le.inverse_transform(y_pred)