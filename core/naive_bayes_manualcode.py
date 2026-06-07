import pandas as pd
import numpy as np


class NaiveBayes:

    def __init__(self):

        # daftar kelas
        self.classes = None

        # prior probability
        self.prior = {}

        # likelihood probability
        self.likelihood = {}

        # daftar nilai fitur
        self.feature_values = {}

    def fit(self, X, y):
        """
        Training model Naive Bayes
        """

        # ubah menjadi DataFrame
        X = pd.DataFrame(X)

        # daftar kelas unik
        self.classes = np.unique(y)

        # total data
        total_data = len(y)

        # ==========================
        # HITUNG PRIOR PROBABILITY
        # ==========================

        for c in self.classes:

            # jumlah data tiap kelas
            class_count = np.sum(y == c)

            # prior probability
            self.prior[c] = class_count / total_data

        # ==========================
        # HITUNG LIKELIHOOD
        # ==========================

        for column in X.columns:

            # simpan semua nilai unik fitur
            self.feature_values[column] = np.unique(
                X[column]
            )

            self.likelihood[column] = {}

            for value in self.feature_values[column]:

                self.likelihood[column][value] = {}

                for c in self.classes:

                    # jumlah data class c
                    class_data = X[y == c]

                    # jumlah fitur=value pada class c
                    value_count = np.sum(
                        class_data[column] == value
                    )

                    # total data class c
                    class_count = len(class_data)

                    # Laplace Smoothing
                    likelihood = (
                        value_count + 1
                    ) / (
                        class_count +
                        len(self.feature_values[column])
                    )

                    self.likelihood[column][value][c] = likelihood

    def predict(self, X):
        """
        Prediksi data
        """

        X = pd.DataFrame(X)

        predictions = []

        for _, row in X.iterrows():

            posteriors = {}

            # hitung posterior tiap kelas
            for c in self.classes:

                posterior = self.prior[c]

                # likelihood tiap fitur
                for column in X.columns:

                    value = row[column]

                    # jika value tidak ada
                    if value not in self.likelihood[column]:

                        likelihood = 1 / (
                            len(X) +
                            len(self.feature_values[column])
                        )

                    else:

                        likelihood = self.likelihood[
                            column
                        ][value].get(c, 1e-6)

                    posterior *= likelihood

                posteriors[c] = posterior

            # pilih probabilitas terbesar
            predicted_class = max(
                posteriors,
                key=posteriors.get
            )

            predictions.append(predicted_class)

        return np.array(predictions)

    def predict_single(self, row):
        """
        Prediksi 1 data + detail probabilitas
        """

        posteriors = {}

        for c in self.classes:

            posterior = self.prior[c]

            for column, value in row.items():

                if value not in self.likelihood[column]:

                    likelihood = 1e-6

                else:

                    likelihood = self.likelihood[
                        column
                    ][value].get(c, 1e-6)

                posterior *= likelihood

            posteriors[c] = posterior

        predicted_class = max(
            posteriors,
            key=posteriors.get
        )

        return predicted_class, posteriors

    def get_prior(self):
        """
        Ambil prior probability
        """

        return self.prior

    def get_likelihood(self):
        """
        Ambil likelihood probability
        """

        return self.likelihood