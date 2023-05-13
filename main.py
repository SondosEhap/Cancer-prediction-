from NbClass import NbModel
from DtreeClass import DtModel
from KnnClass import knnModel
from LogClass import logModel
import pandas as pd
#from sklearn import svm
#from sklearn.svm import SVC
import joblib
import matplotlib.pyplot as plt

data2 = pd.read_csv(r"data3.csv")
data2 = data2[['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15','F16','F17','F18','F19','F20','F21','F22','F23','F24','F25','F26','F27','F28','F29','F30']]

#Drop the rows that contain missing values.
data2.dropna(how='any',inplace=True, axis=0)

data2.T.plot()
plt.ylabel('diagnosis')
class main:
    
    log=logModel()
    nb=NbModel()
    dTree=DtModel()
    knn=knnModel()
    
# load the logistic regression model from disk
    loaded_model = joblib.load(log.filename)
    result1 = loaded_model.predict(data2)
    print("LOGISTIC REGRESSION : ",result1)

# load the decision tree model from disk
    loaded_model2 = joblib.load(dTree.filename2)
    result2 = loaded_model2.predict(data2)
    print("DECISION TREE : ",result2)

# load the KNN model from disk
    loaded_model3 = joblib.load(knn.filename3)
    result3 = loaded_model3.predict(data2)
    print("KNN : ",result3)

# load the SVM model from disk
        #loaded_model5 = joblib.load(filename5)
        #result5 = loaded_model5.predict(data2)
        #print("SVM : ",result5)

# load the naive bayes model from disk
    loaded_model4 = joblib.load(nb.filename4)
    result4 = loaded_model4.predict(data2)
    print("naive bayes : ",result4)
    
    def voting_classifier(result1,result2, result3):
        
        for x,y,z in zip(result1,result2, result3):
            if (x==1 and y==1) or (x==1 and z==1) or (z==1 and y==1):
                print("1") 
            else: print("0")  

    voting_classifier(result1,result2, result3)
