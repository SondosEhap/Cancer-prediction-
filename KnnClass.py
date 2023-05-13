from sklearn.neighbors import KNeighborsClassifier
import NbClass as fn
import joblib
from sklearn import metrics

class knnModel:
#knn model
    model3=KNeighborsClassifier(n_neighbors=3)
    z_predict=fn.myFunction(model3)
# save the model to disk
    filename3 = 'finalized_model3.joblib'
    joblib.dump(model3, filename3)
    print("KNN Model")
    print("Accuracy:",metrics.accuracy_score(fn.y_test, z_predict))
    print("Precision:",metrics.precision_score(fn.y_test, z_predict))
    print("Recall:",metrics.recall_score(fn.y_test,z_predict))