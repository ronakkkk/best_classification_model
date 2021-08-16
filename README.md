Best Classification Model is used for supervised learning techniques where the target data is in binary form. It selects the best model from the seven classification model based on the accuracy. 

The seven classification model used in the given assignment are:

1. Logistic Regression
2. Naive Bayes
3. Stochastic Gradient Classifier
4. K Neighbors Classifier
5. Decision Tree Classifier
6. Random Forest Classifier
7. Support Vector Machine

#### User installation

If you already have a working installation of numpy, scipy and sklearn, the easiest way to install best-classification-model is using pip

#### `pip install BestClassificationModel`

#### Important links

Official source code repo: https://github.com/ronakkkk/best_classification_model

Download releases: https://pypi.org/project/BestClassificationModel/

#### Examples
```import

from Best_Classification_Model import best_model

import pandas

data = pandas.read_csv('Data.csv')

X = data.iloc[:, :-1]

Y = data['Class']

best_model, best_model_name, acc = best_model.bestClassificationModel(X, Y)

print(best_model)

print(best_model_name, ":", acc)```

`__Output__:
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=None, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=10,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)

Random Forest:0.861145`

 
