from sklearn.linear_model import LogisticRegression
import NbClass as fn
import joblib
from sklearn import metrics
import numpy as np


class logModel:
#logisticregression model
    model = LogisticRegression()
    z_predict=fn.myFunction(model)
# save the model to disk
    filename='finalized_model.joblib'
    joblib.dump(model, filename)
    cnf_matrix = metrics.confusion_matrix(fn.y_test, z_predict)
    print("LogisticRegression Model")
    print("Accuracy:",metrics.accuracy_score(fn.y_test, z_predict))
    print("Mean Square Error: ", metrics.mean_squared_error(np.asarray(fn.y_test), z_predict))
    print("Precision:",metrics.precision_score(fn.y_test, z_predict))
    print("Recall:",metrics.recall_score(fn.y_test, z_predict))

