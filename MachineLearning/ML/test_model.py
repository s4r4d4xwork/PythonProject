from joblib import load
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    #test the trained model
    final_model = load('D:\FPT\py-project\ML\output\music_rcm.joblib')
    # data of row 1 in data.csv
    predictions1 = final_model.predict([[0.803,0.624,7,-6.764,0,0.0477,0.451,0.000734,0.1,0.628,95.968,304524,4]])
    # data of row 2 in data.csv
    predictions2 = final_model.predict([[0.762,0.703,10,-7.951,0,0.306,0.206,0.0,0.0912,0.519,151.329,247178,4]])


print('Prediction1: ')
if(predictions1[0] == 1):
    print('Liked')
else:
    print('Disliked')

print('Prediction2: ')
if(predictions2[0] == 1):
    print('Liked')
else:
    print('Disliked')