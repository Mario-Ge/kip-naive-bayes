import joblib
import numpy as np

# load model
model = joblib.load("model/model.pkl")

# load preprocessor
prep = joblib.load(
    "model/preprocessor.pkl"
)

print("=" * 60)
print("MODEL NAIVE BAYES")
print("=" * 60)

# =====================================
# CLASSES
# =====================================

print("\nCLASSES:")

for i, c in enumerate(model.model.classes_):

    label = prep.encoders["target"].inverse_transform(
        [c]
    )[0]

    print(f"{c} -> {label}")

# =====================================
# PRIOR
# =====================================

print("\nPRIOR PROBABILITY:")

prior_probs = np.exp(
    model.model.class_log_prior_
)

for i, prob in enumerate(prior_probs):

    label = prep.encoders["target"].inverse_transform(
        [i]
    )[0]

    print(f"{label} -> {prob}")

# =====================================
# LIKELIHOOD
# =====================================

print("\nLIKELIHOOD PROBABILITY:")

features = list(
    prep.encoders.keys()
)

features.remove("target")

for feature_index, feature_name in enumerate(features):

    print(f"\nFitur: {feature_name}")

    encoder = prep.encoders[feature_name]

    categories = encoder.classes_

    for class_index, class_probs in enumerate(
        model.model.feature_log_prob_[feature_index]
    ):

        target_label = prep.encoders[
            "target"
        ].inverse_transform(
            [class_index]
        )[0]

        print(f"\nClass: {target_label}")

        probs = np.exp(class_probs)

        for category_index, prob in enumerate(probs):

            category_name = categories[
                category_index
            ]

            print(
                f"{category_name} -> {prob}"
            )

print("\n" + "=" * 60)