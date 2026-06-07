from sklearn.naive_bayes import CategoricalNB
import numpy as np


class NaiveBayes:

    def __init__(self):

        # inisialisasi model
        self.model = CategoricalNB()

    def fit(self, X, y):
        """
        Training model Naive Bayes
        """

        self.model.fit(X, y)

    def predict(self, X):
        """
        Prediksi banyak data
        """

        return self.model.predict(X)

    def predict_single(self, row):
        """
        Prediksi 1 data
        """

        row_array = np.array(row).reshape(1, -1)

        prediction = self.model.predict(row_array)

        probabilities = self.model.predict_proba(
            row_array
        )

        return prediction[0], probabilities[0]

    def predict_proba(self, X):
        """
        Ambil probabilitas prediksi
        """

        return self.model.predict_proba(X)

    def get_classes(self):
        """
        Ambil daftar class
        """

        return self.model.classes_

    def get_prior(self):
        """
        Ambil prior probability
        """

        return self.model.class_log_prior_

    def get_feature_log_prob(self):
        """
        Ambil likelihood probability
        """

        return self.model.feature_log_prob_