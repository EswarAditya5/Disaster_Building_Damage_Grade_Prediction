import joblib
def predict(data):
    clf = joblib.load('tree_model_smote.sav')
    return clf.predict(data)
