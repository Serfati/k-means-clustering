# Import models from scikit learn module:
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import KFold  # For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
import numpy as np


# Generic function for making a classification model and accessing performance:
def classification_model(_model, data, predictors, outcome):
    # Fit the model:
    _model.fit(data[predictors], data[outcome])

    # Make predictions on training set:
    predictions = _model.predict(data[predictors])

    # Print accuracy
    accuracy = metrics.accuracy_score(predictions, data[outcome])
    print("Training accuracy : %s" % "{0:.3%}".format(accuracy))

    # Perform k-fold cross-validation with 5 folds
    kf = KFold(n_splits=5).split()
    accuracy = []
    for train, test in kf:
        # Filter training data
        train_predictors = (data[predictors].iloc[train, :])

        # The target we're using to train the algorithm.
        train_target = data[outcome].iloc[train]

        # Training the algorithm using the predictors and target.
        _model.fit(train_predictors, train_target)

        # Record accuracy from each cross-validation run
        accuracy.append(_model.score(data[predictors].iloc[test, :], data[outcome].iloc[test]))

    print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(accuracy)))

    # Fit the model again so that it can be refered outside the function:
    _model.fit(data[predictors], data[outcome])


def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.
    Open the generate 'tree.dot' file in notepad and copy its contents to http://webgraphviz.com/.
    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    dotfile = 'tree.dot'
    export_graphviz(tree, out_file=dotfile, feature_names=feature_names, class_names=['No', 'Yes'])


# Reading the dataset in a dataframe using Pandas
df = pd.read_csv("/home/serfati/Desktop/DS/Labs/lab_9/train_Loan.csv")
var_mod = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
le = LabelEncoder()

for i in var_mod:
    df[i] = le.fit_transform(df[i])
df = df.fillna('0')
print(df)

outcome_var = 'Loan_Status'
predictor_var = ['Credit_History', 'Gender', 'Married', 'Education']
myModel = DecisionTreeClassifier()
classification_model(myModel, df, predictor_var, outcome_var)

# We can try different combination of variables:
predictor_var = ['Credit_History', 'Loan_Amount_Term', 'LoanAmount']
classification_model(myModel, df, predictor_var, outcome_var)

visualize_tree(myModel, predictor_var)

# Applying Random Forest Algorithm
model = RandomForestClassifier(n_estimators=100)
predictor_var = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History',
                 'Property_Area', 'LoanAmount']
classification_model(model, df, predictor_var, outcome_var)

# Create a series with feature importances:
featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
print(featimp)

# Apply Random Forest with improved predictors and parameters
model = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features='auto')
predictor_var = ['LoanAmount', 'Credit_History', 'Dependents', 'Property_Area']
classification_model(model, df, predictor_var, outcome_var)

visualize_tree(myModel, predictor_var)
