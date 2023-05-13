from sklearn.naive_bayes import GaussianNB
import joblib
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns 

data = pd.read_csv(r"C:\Users\Alrahma\\Downloads\Tumor Cancer Prediction_Data.csv")

#Drop the rows that contain missing values.
data.dropna(how='any',inplace=True, axis=0)
data = data.drop("Index", axis=1)

#converting data to 0,1
data=data.replace(to_replace=['B','M'],value=[0,1])

#Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('diagnosis',axis=1),data['diagnosis'], test_size = 0.25, random_state=101)



sns.countplot(data['diagnosis'], label='count')


#heat plot
#sns.heatmap(data.corr()) #heatmap

def myFunction(model):
 model.fit(X_train, y_train)
 y_predict=model.predict(X_test)   
 return y_predict


class NbModel:
#naive_bayes model
    model4 =GaussianNB()
    z_predict=myFunction(model4)
# save the model to disk
    filename4 = 'finalized_model4.joblib'
    joblib.dump(model4, filename4)
    print("Naive Bayes Model")
    print("Accuracy:",metrics.accuracy_score(y_test,z_predict))

