import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score,roc_auc_score
Bank_det = pd.read_excel('C:/Users/Dinakar/Downloads/Bankdataclean.xlsx')
Bank_det
Bank_det['CHURN'].value_counts()
X = Bank_det.loc[:,['AGE', 'CUS_Month_Income', 'YEARS_WITH_US']]
X.shape
Y = pd.DataFrame(Bank_det.loc[:,"CHURN"])
Y.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y , test_size = 0.2, random_state = 1234)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
model = LogisticRegression(max_iter=2000,class_weight={0:1,1:5})
model.fit(X_train,np.ravel(y_train))
model.get_params()
X_test = scaler.fit_transform(X_test)
y_pred = model.predict(X_test)
print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test,y_pred))
from sklearn.metrics import roc_curve, auc
logistic_fpr, logistic_tpr, threshold = roc_curve(y_test, y_pred)
auc_logistic = auc(logistic_fpr, logistic_tpr)


plt.plot(logistic_fpr, logistic_tpr, marker='.', label='Logistic (auc = %0.3f)' % auc_logistic)

plt.xlabel('False Positive Rate (1-specificity) -->')
plt.ylabel('True Positive Rate (Recall) -->')
plt.legend()
plt.show()
import pickle
pickle.dump( model, open( 'model.pkl', 'wb' ) )