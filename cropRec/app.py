from flask import Flask,render_template,request
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb


con=mysql.connector.connect(user="root",password="8439",database="crop");
cur=con.cursor()

app=Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/adminlogin")
def adminlogin():
    return render_template("adminlogin.html")


@app.route("/farmerlogin")
def farmerlogin():
    return render_template("farmerlogin.html")


@app.route("/farmerreg")
def farmerreg():
    return render_template("farmerreg.html")


@app.route("/train")
def train():
    global X_train, X_test, y_train, y_test,model
    # Reading the Dataset
    cropdf = pd.read_csv("crop.csv")

    # Print some sample data from dataset
    cropdf.head()
    mean_value=cropdf['N'].mean()
    cropdf['N'].fillna(value=mean_value, inplace=True)

    mean_value=cropdf['P'].mean()
    cropdf['P'].fillna(value=mean_value, inplace=True)

    mean_value=cropdf['K'].mean()
    cropdf['K'].fillna(value=mean_value, inplace=True)

    mean_value=cropdf['temperature'].mean()
    cropdf['temperature'].fillna(value=mean_value, inplace=True)

    mean_value=cropdf['humidity'].mean()
    cropdf['humidity'].fillna(value=mean_value, inplace=True)

    mean_value=cropdf['ph'].mean()
    cropdf['ph'].fillna(value=mean_value, inplace=True)

    mean_value=cropdf['rainfall'].mean()
    cropdf['rainfall'].fillna(value=mean_value, inplace=True)

    mean_value=cropdf['Area'].mean()
    cropdf['Area'].fillna(value=mean_value, inplace=True)

    mean_value=cropdf['Production'].mean()
    cropdf['Production'].fillna(value=mean_value, inplace=True)

    cropdf["Season"].fillna("Kharif", inplace = True)
    cropdf['Season']=cropdf['Season'].str.strip()
    cropdf['Season'] = cropdf['Season'].replace('Kharif',1)
    cropdf['Season'] = cropdf['Season'].replace('Autumn',2)
    cropdf['Season'] = cropdf['Season'].replace('Rabi',3)
    cropdf['Season'] = cropdf['Season'].replace('summer',4)
    cropdf['Season'] = cropdf['Season'].replace('winter',5)
    cropdf['Season'] = cropdf['Season'].replace('Whole Year',6)
    print("Dataset is ")
    print(cropdf.head())

    cropdf.reset_index()
    # Extract only parametres and drop end results
    X = cropdf.drop('label', axis=1)

    # Extract only end results 
    y = cropdf['label']


    # import library to split training and testing data
    from sklearn.model_selection import train_test_split

    # X_train - Training Dataset
    # y_train - Expected Training Results

    # X_test - Testing Dataset
    # y_test - Expected Testing Results

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                        shuffle = True, random_state = 0)



    model = lgb.LGBMClassifier()

    # Training the model using Training Data
    model.fit(X_train, y_train)
    filename = 'cropmodel.pkl'
    pickle.dump(model, open(filename, 'wb'))

    return render_template("train.html")

@app.route("/accuracy")
def accuracy():
    y_pred=model.predict(X_test)
    # Library to measure accuracy of model
    from sklearn.metrics import accuracy_score

    # Find accuracy on Expected Output and Predicted Output on Testing Data
    accuracy=accuracy_score(y_pred, y_test)
    print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
    return render_template("accuracy.html",acc=accuracy)


@app.route("/recommend")
def recommend():
    return render_template("predict.html")



@app.route("/predictCrop",methods=['POST'])
def predictCrop():
    v1=request.form['t1']
    v2=request.form['t2']
    v3=request.form['t3']
    v4=request.form['t4']
    v5=request.form['t5']
    v6=request.form['t6']
    v7=request.form['t7']
    v8=request.form['t8']
    v9=request.form['t9']
    v10=request.form['t10']
    filename = 'cropmodel.pkl'
    model = pickle.load(open(filename, 'rb'))
    sea=0
    if v9=='Kharif':
        sea=1
    elif v9=='Autumn':
        sea=2
    elif v9=='Rabi':
        sea=3
    elif v9=='summer':
        sea=4
    elif v9=='winter':
        sea=5
    elif v9=='Whole Year':
        sea=6

    x=model.predict([[v1, v2, v3, v4, v5, v6,v7, v8, float(sea),v10]])
    print(x)

    return render_template("predictcrop.html",rc=x)


@app.route("/viewdataset")
def viewdataset():
    from csv import reader
    lst=[]

    with open("crop.csv","r", encoding = 'unicode_escape') as obj:
        csv_reader=reader(obj)
        lst=list(csv_reader)

    return render_template("viewdataset.html",data=lst)


@app.route("/adminloginDB",methods=['POST'])
def adminloginDB():
    uname=request.form['un']
    pwd=request.form['pwd']
    if uname=='diet' and pwd=='diet':
        return render_template("adminhome.html")
    else:
        return render_template("adminlogin.html",msg="Pls Check the Credentials")



@app.route("/regDB",methods=['POST'])
def regDB():
    name=request.form['name']
    uname=request.form['un']
    pwd=request.form['pwd']
    aadhar=request.form['aadhar']
    contact=request.form['contact']
    s="insert into farmer(name,username,password,aadhar,contact) values('"+name+"','"+uname+"','"+pwd+"','"+aadhar+"','"+contact+"')"    
    cur.execute(s)
    con.commit()
    return render_template("farmerlogin.html",msg="")

@app.route("/farmerloginDB",methods=['POST'])
def farmerloginDB():
    uname=request.form['un']
    pwd=request.form['pwd']
    s="select * from farmer where username='"+uname+"' and password='"+pwd+"'"
    cur.execute(s)
    data=cur.fetchall()
    if len(data)>0:
        return render_template("farmerhome.html")
    else:
        return render_template("farmerlogin.html",msg="Pls Check the Credentials")



@app.route("/logout")
def logout():
    return render_template("index.html")


app.run(debug=True)
