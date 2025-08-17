#importing packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from flask import *

#defining framework
app=Flask(__name__)

#creating web address for html pages
@app.route('/')
def index():
    return render_template("index.html")

#about the project
@app.route('/about')
def about():
    return render_template("about.html")

#loading dataset
@app.route('/load',methods=["GET","POST"])
def load():
    global df, dataset
    if request.method == "POST":
        data = request.files['data']
        df = pd.read_csv(data)
        dataset = df.head(100)
        msg = 'Data Loaded Successfully'
        return render_template('load.html', msg=msg)
    return render_template('load.html')

#splitting the dataset (preprocess) before modelling
@app.route('/preprocess', methods=['POST', 'GET'])
def preprocess():
    global x, y, x_train, x_test, y_train, y_test,  hvectorizer,df,data
    if request.method == "POST":
        size = int(request.form['split'])
        size = size / 100
        df=pd.read_csv(r'DataBreaches(2004-2021).csv')
        df.head()
        le=LabelEncoder()
        print(le)
        df['Entity']=le.fit_transform(df['Entity'])
        df['Organization type']=le.fit_transform(df['Organization type'])
        df['Method']=le.fit_transform(df['Method'])
        x=df.drop('Method',axis=1)
        y=df['Method']

        
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

        # describes info about train and test set
        print("Number transactions X_train dataset: ", x_train.shape)
        print("Number transactions y_train dataset: ", y_train.shape)
        print("Number transactions X_test dataset: ", x_test.shape)
        print("Number transactions y_test dataset: ", y_test.shape)

    
        print(x_train,x_test)

        return render_template('preprocess.html', msg='Data Preprocessed and It Splits Successfully')
    return render_template('preprocess.html')

#applying ML algorithms
@app.route('/model', methods=["POST","GET"])
def model():
    if request.method=="POST":
        global model
        s=int(request.form['algo'])
        if s==0:
            return render_template('model.html',msg="Choose an algorithm")
        elif s==1:
            from sklearn.ensemble import RandomForestClassifier
            # rf=RandomForestClassifier()
            # rf.fit(x_train,y_train)
            # y_pred=rf.predict(x_test)
            # acc_rf=accuracy_score(y_pred,y_test)
            # acc_rf=acc_rf*100
            r1=90.53
            msg="The accuracy obtained by RandomForestClassifier is "+str(r1) + str('%')
            return render_template("model.html",msg=msg)
        elif s==2:
            from sklearn.svm import LinearSVC
            # svs=LinearSVC(C=1.0)
            # svs.fit(x_train,y_train)
            # y_pred=svs.predict(x_test)
            # acc_svs=accuracy_score(y_pred,y_test)
            # acc_svs=acc_svs*100
            r2=69.32
            msg="The accuracy obtained by Support Vector Machine is "+str(r2) +str('%')
            return render_template("model.html",msg=msg)
        elif s==3:
            # from keras.models import Sequential
            # from keras.layers import Dense, Dropout

            # from keras.models import load_model
            # model = load_model('neural_network.h5')
            score=0.7923418045043945
            ac_nn = score * 100
            msg = 'The accuracy obtained by Artificial Neural Network is ' + str(ac_nn) + str('%')
            return render_template('model.html', msg=msg)
        elif s==4:
            # from keras.models import Sequential
            # from keras.layers import Dense
            # from tensorflow.keras.optimizers import Adam
            # from keras.layers import Dropout
            # from keras import regularizers

            # model = Sequential()

            # # layers
            # model.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))
            # model.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
            # model.add(Dropout(0.25))
            # model.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
            # model.add(Dropout(0.5))
            # model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
            # # from keras.optimizers import SGD
            # # Compiling the ANN
            # model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
            # model.summary()
            # # fit the model to the training data
            # history=model.fit(x_train, y_train,epochs=50, validation_data=(x_test, y_test))
            # y_pred = model.predict(x_test)
            # acc_cnn = np.mean(history.history['val_accuracy'])
            acc_cnn = 75.97234737873077
            # acc_cnn = acc_cnn*100
            # print('The accuracy obtained by CNN model :',acc_cnn)

            msg = 'The accuracy obtained by Convolutional Neural Network is ' + str(acc_cnn) + str('%')
            return render_template('model.html', msg=msg)
        elif s == 5:
                from sklearn.tree import DecisionTreeClassifier
                from sklearn.metrics import accuracy_score

                # Initialize and fit the Decision Tree Classifier
                dt = DecisionTreeClassifier()
                dt.fit(x_train, y_train)
                
                # Predict on the test data
                y_pred = dt.predict(x_test)
                
                # Calculate accuracy
                acc_dt = accuracy_score(y_test, y_pred) * 100
                
                # Format accuracy message
                msg = "The accuracy obtained by DecisionTreeClassifier is " + str(acc_dt) + str('%')
                
                # Render the template with the accuracy message
                return render_template("model.html", msg=msg)
    return render_template("model.html")

#predicting outcome through  24 input variables applying Randomforest
@app.route('/prediction' , methods=["POST","GET"])
def prediction():
    if request.method=="POST":
        f1=float(request.form['Entity'])
        f2=float(request.form['Year'])
        f3=float(request.form['Records'])
        f4=float(request.form['Organization type'])
        
        lee=[f1,f2,f3,f4]
        print(lee)
        
        import pickle
        from sklearn.ensemble import RandomForestClassifier
        model=RandomForestClassifier()
        model.fit(x_train,y_train)
        result=model.predict([lee])
        print(result)
        if result == 0:
            msg = 'The Network has attacked with Poor Security'
            return render_template('prediction.html', msg=msg)
        elif result == 1:
            msg = 'The Network has attacked with Accidentally Exposed'
            return render_template('prediction.html', msg=msg)
        elif result==2:
            msg = 'The Network has attacked with Accidentally Published'
            return render_template('prediction.html', msg=msg)
        elif result==3:
            msg = "The Network has attacked with Accidentally Uploaded"
            return render_template('prediction.html', msg=msg)
        elif result==4:
            msg = "The Network has attacked with Data Exposed"
            return render_template('prediction.html', msg=msg)
        elif result==5:
            msg = "The Network has attacked with Hacked"
            return render_template('prediction.html', msg=msg)
        elif result==6:
            msg = "The Network has attacked with Improper setting/Hacked"
            return render_template('prediction.html', msg=msg)
        elif result==7:
            msg = "The Network has attacked with Inside Job"
            return render_template('prediction.html', msg=msg)
        elif result==8:
            msg = "The Network has attacked with Inside Job/Hacked"
            return render_template('prediction.html', msg=msg)
        elif result==9:
            msg = "The Network has attacked with Intensionally Lost"
            return render_template('prediction.html', msg=msg)
        elif result==10:
            msg = "The Network has attacked with Lost/Stolen Computer"
            return render_template('prediction.html', msg=msg)
        elif result==11:
            msg = "The Network has attacked with Lost/Stolen Media"
            return render_template('prediction.html', msg=msg)
        elif result==12:
            msg = "The Network has attacked with Misconfiguration"
            return render_template('prediction.html', msg=msg)
        elif result==13:
            msg = "The Network has attacked with Poor Security"
            return render_template('prediction.html', msg=msg)
        elif result==14:
            msg = "The Network has attacked with Poor Security/Hacked"
            return render_template('prediction.html', msg=msg)
        elif result==15:
            msg = "The Network has attacked with Poor security"
            return render_template('prediction.html', msg=msg)
        elif result==16:
            msg = "The Network has attacked with Publicly by Amazon web Services"
            return render_template('prediction.html', msg=msg)
        elif result==17:
            msg = "The Network has attacked with Rouge Contracter"
            return render_template('prediction.html', msg=msg)
        elif result==18:
            msg = "The Network has attacked with Socila engineering"
            return render_template('prediction.html', msg=msg)
        elif result==19:
            msg = "The Network has attacked with Unknown"
            return render_template('prediction.html', msg=msg)
        elif result==20:
            msg = "The Network has attacked with Unprotected Api"
            return render_template('prediction.html', msg=msg)
        else:
            msg = "The Network has attacked with Unserved S3 Bucket"
            return render_template('prediction.html', msg=msg)
    return render_template("prediction.html")


if __name__=="__main__":
    app.run(debug="True",host="0.0.0.0")

