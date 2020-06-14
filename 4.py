import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold, cross_val_score  # For K-fold CV
from sklearn.preprocessing import LabelEncoder


# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(6, input_dim=59, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(2, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# load dataset
dataframe = pandas.read_csv("sonar.csv", header=None)
dataset = dataframe.values
X = dataset[:, 0:59].astype(float)
Y = dataset[:, 60]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

estimator = KerasClassifier(build_fn=baseline_model, epochs=100, batch_size=5, verbose=1)
skf = StratifiedKFold(n_splits=10)
results = cross_val_score(estimator=estimator, X=X, y=dummy_y, cv=skf.get_n_splits(X.shape[0], dummy_y))
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# Baseline: 61.76% (23.71%)
