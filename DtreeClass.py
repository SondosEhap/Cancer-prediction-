from sklearn.tree import DecisionTreeClassifier
import NbClass as fn
import joblib
from sklearn import metrics


class DtModel:
#decision tree model
    model2 =DecisionTreeClassifier(max_depth=10,random_state=101,max_features=None,min_samples_leaf=15)
    z_predict=fn.myFunction(model2)
# save the model to disk
    filename2 = 'finalized_model2.joblib'
    joblib.dump(model2, filename2)
    print("Decision Tree Model")
    print("Accuracy:",metrics.accuracy_score(fn.y_test,z_predict))