import pickle
import pandas as pd;
from sklearn.model_selection import train_test_split;
from sklearn.linear_model import LogisticRegression;

data = pd.read_csv('diabetes.csv')


data_c = data[(data.BloodPressure != 0) & (data.BMI != 0) & (data.Glucose) !=0];

feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'];

X = data[feature_names];
y = data.Outcome;

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0);

logreg = LogisticRegression();
logreg.fit(X_train,y_train);


pkl_filename = "pickle_model.pkl"  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(logreg, file)

# Load from file
with open(pkl_filename, 'rb') as file:  
    pickle_model = pickle.load(file)

# Calculate the accuracy score and predict target values
score = pickle_model.score(X_test, y_test)  
print("Test score: {0:.2f} %".format(100 * score))  
Ypredict = pickle_model.predict(X_test)  



