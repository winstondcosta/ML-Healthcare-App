import streamlit as st
st. set_page_config(layout="wide", page_icon=":hospital:")
st.set_option('deprecation.showPyplotGlobalUse', False)
import pandas as pd
import numpy as np
import seaborn as sns
import time
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.style.use('dark_background')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.metrics import precision_recall_fscore_support as score, mean_squared_error
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import os
import shutil
import random
import torch
import torchvision
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
torch.manual_seed(0)
import matplotlib.image as mpimg
#%matplotlib inline
from utils import preproces, CoughNet
#import tensorflow.keras.models as models
#--------------------------------------------------------------------------------------------------------------------------------------------------------
start_time=time.time()  #Program Start time

left, centre, right = st.columns([1, 2, 1])
with left:
    st.image("cirrusrays_logo.jpeg", width=150)
with centre:
    st.markdown("<h1 style='text-align: center;'><u>HealthCare App</u> </h1>",unsafe_allow_html=True)
    style = "<style>h2 {text-align: center;}</style>"
    st.markdown(style, unsafe_allow_html=True)

# style = "<style>h2 {text-align: center;}</style>"
# st.markdown(style, unsafe_allow_html=True)
# st.image("cirrusrays_logo.jpeg")
# st.columns(3)[1].header("hello world")
# tit2.image("cirrusrays_logo.jpeg")

st.sidebar.title("Dataset and Classifier")
dataset_name = st.sidebar.selectbox("Select the Task: ",('Heart Attack',"Breast Cancer","Skin Cancer","Covid-19"))
train_inf = st.sidebar.radio("Do you want to train or use pretrained model", ["train", "Inference"])

if train_inf == 'train':
    if dataset_name == 'Heart Attack' or dataset_name == 'Breast Cancer':
        classifier_name = st.sidebar.selectbox("Select Classifier: ",("Logistic Regression","KNN","SVM","Decision Trees","Random Forest","Gradient Boosting","XGBoost"))
    else:
        classifier_name = st.sidebar.selectbox("Select Classifier: ",("CNN", "RNN"))
else:
    model_path = st.file_uploader("Choose a h5 file", type=["h5", "pth"])
    if model_path is not None:
        if model_path.name[-3:] == 'pth':
            checkpoint = torch.load(model_path)
            #st.write(checkpoint)
            clf = CoughNet(len(checkpoint['hparams']['features']))
            clf.load_state_dict(checkpoint['model_state'])
        else:
            clf = torch.load(model_path)
        #st.write(clf)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_dirs, transform, dataset):
        def get_images(class_name):
            images = [x for x in os.listdir(image_dirs[class_name]) if x[-3:].lower().endswith('png') or x[-3:].lower().endswith('jpg')]
            print(f'Found {len(images)} {class_name} examples')
            return images
        
        self.images = {}
        if dataset == 'Covid-19':
            self.class_names = ['normal', 'viral', 'covid']
        elif dataset == 'Skin Cancer':
            self.class_names = ['benign', 'malignant']
        
        for class_name in self.class_names:
            self.images[class_name] = get_images(class_name)
            
        self.image_dirs = image_dirs
        self.transform = transform
        
    
    def __len__(self):
        return sum([len(self.images[class_name]) for class_name in self.class_names])
    
    
    def __getitem__(self, index):
        class_name = random.choice(self.class_names)
        index = index % len(self.images[class_name])
        image_name = self.images[class_name][index]
        image_path = os.path.join(self.image_dirs[class_name], image_name)
        image = Image.open(image_path).convert('RGB')
        return self.transform(image), self.class_names.index(class_name)

display_train_transform = transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224)),
    transforms.ToTensor()
])

train_transform = transforms.Compose([
    #torchvision.transforms.Resize(size=(224, 224)),
    transforms.Resize(size=(32,32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    #transforms.Resize(size=(224, 224)),
    transforms.Resize(size=(32,32)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

hparams = {    
                'dataset': 'Data/prepared_data_balanced.csv',
                'epochs': 1,
                'batch_size': 16,
                'lr': 1e-3,
                'features': [
                    'chroma_stft', 'rmse', 'spectral_centroid', 'spectral_bandwidth', 'rolloff', 'zero_crossing_rate',
                    'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 
                    'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20'
                ]
            }

## Code to display X-ray images
def show_images(images, labels, preds):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, file in enumerate(images[:6]):
        print(file.shape)
        image = file.numpy().transpose((1, 2, 0))

        # Read the uploaded image using Matplotlib
        # img = mpimg.imread(file)
        # Determine subplot position
        row = i // 3
        col = i % 3
        # Display the image on the corresponding subplot
        axes[row, col].imshow(image)
        axes[row, col].axis('off')  # Turn off axis
    # Adjust layout
    plt.tight_layout()
    # Show the plot in Streamlit
    st.pyplot(fig)

flag = False

LE=LabelEncoder()

def get_ml_dataset(dataset_name):
    if dataset_name=="Heart Attack":
        data=pd.read_csv("https://raw.githubusercontent.com/advikmaniar/ML-Healthcare-Web-App/main/Data/heart.csv")
        st.header("Heart Attack Prediction")
        return data

    elif dataset_name=="Breast Cancer":
        data=pd.read_csv("https://raw.githubusercontent.com/advikmaniar/ML-Healthcare-Web-App/main/Data/BreastCancer.csv")
        
        data["diagnosis"] = LE.fit_transform(data["diagnosis"])
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data["diagnosis"] = pd.to_numeric(data["diagnosis"], errors="coerce")
        st.header("Breast Cancer Prediction")
        return data

def get_dataset(dataset_name): 
    if dataset_name=="Skin Cancer":
        data_type = 'Image'
        st.header("Skin Cancer Prediction")
        classes = 2
        train_dirs = {
            'benign': 'Data/cancer/train/benign',
            'malignant': 'Data/cancer/train/malignant',
        }

        test_dirs = {
            'benign': 'Data/cancer/test/benign',
            'malignant': 'Data/cancer/test/malignant',
        }
        train = Dataset(train_dirs, display_train_transform, dataset_name)
        train_data = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
        images, labels = next(iter(train_data))
        show_images(images, labels, labels)

        train_dataset = Dataset(train_dirs, train_transform, dataset_name)
        val_dataset = Dataset(test_dirs, test_transform, dataset_name)

        train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_dataset = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=True)
        return train_dataset, val_dataset, classes, data_type
    	 
    elif dataset_name=="Covid-19":
        st.header("Covid-19 Prediction")
        data_type = st.sidebar.selectbox("Select the data type: ",('Image','Audio'))
        if data_type == 'Image':
            classes = 3
            train_dirs = {
                'normal': 'Data/COVID-19_Radiography_Database/normal',
                'viral': 'Data/COVID-19_Radiography_Database/viral',
                'covid': 'Data/COVID-19_Radiography_Database/covid'
            }

            test_dirs = {
                'normal': 'Data/COVID-19_Radiography_Database/test/normal',
                'viral': 'Data/COVID-19_Radiography_Database/test/viral',
                'covid': 'Data/COVID-19_Radiography_Database/test/covid'
            }	

            train = Dataset(train_dirs, display_train_transform, dataset_name)
            train_data = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
            images, labels = next(iter(train_data))
            show_images(images, labels, labels)

            train_dataset = Dataset(train_dirs, train_transform, dataset_name)
            val_dataset = Dataset(test_dirs, test_transform, dataset_name)
            
            train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
            val_dataset = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=True)
            return train_dataset, val_dataset, classes, data_type

        elif data_type == 'Audio':
            classes = 2

            df_features = pd.read_csv(hparams['dataset'])

            X = np.array(df_features[hparams['features']], dtype=np.float32)

            encoder = LabelEncoder()
            y = encoder.fit_transform(df_features['label'])
            print('classes:', encoder.classes_)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # scale data
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            print('X_train.shape:', X_train.shape)
            print('y_train.shape:', y_train.shape)

            # create pytorch dataloader
            torch.manual_seed(42)
            train_dataset = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train).long())
            test_dataset = torch.utils.data.TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test).long())
            train_dataset = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=hparams['batch_size'], shuffle=True)
            val_dataset = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=hparams['batch_size'], shuffle=False)

            checkpoint = {
                'scaler': scaler,
                'encoder': encoder
                }
            
            torch.save(checkpoint, 'checkpoint.pth')

            return train_dataset, val_dataset, classes, data_type

if dataset_name == 'Covid-19' or dataset_name == 'Skin Cancer':
    train_dataset, val_dataset, classes, data_type = get_dataset(dataset_name)
else:
    data = get_ml_dataset(dataset_name)

  
def selected_dataset(dataset_name):
    if dataset_name == "Heart Attack":
        X=data.drop(["output"],axis=1)
        Y=data.output
        return X,Y

    elif dataset_name == "Breast Cancer":
        X = data.drop(["id","diagnosis"], axis=1)
        Y = data.diagnosis
        return X,Y
    	
if dataset_name == "Heart Attack" or dataset_name == "Breast Cancer":
	X, Y = selected_dataset(dataset_name)


#Plot output variable
def plot_op(dataset_name):
    col1, col2 = st.beta_columns((1, 5))
    plt.figure(figsize=(12, 3))
    plt.title("Classes in 'Y'")
    if dataset_name == "Heart Attack":
        col1.write(Y)
        sns.countplot(Y, palette='gist_heat')
        col2.pyplot()

    elif dataset_name == "Breast Cancer":
        col1.write(Y)
        sns.countplot(Y, palette='gist_heat')
        col2.pyplot()

if dataset_name == "Heart Attack" or dataset_name == "Breast Cancer":
	st.write(data)
	st.write("Shape of dataset: ",data.shape)
	st.write("Number of classes: ",Y.nunique())
	plot_op(dataset_name)
	
# elif dataset_name == "Covid-19" or dataset_name == "Skin Cancer":
# 	train_data = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
# 	test_data = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

def add_parameter_ui(clf_name):
    params={}
    st.sidebar.write("Select values: ")

    if clf_name == "Logistic Regression":
        R = st.sidebar.slider("Regularization",0.1,10.0,step=0.1)
        MI = st.sidebar.slider("max_iter",50,400,step=50)
        params["R"] = R
        params["MI"] = MI

    elif clf_name == "KNN":
        K = st.sidebar.slider("n_neighbors",1,20)
        params["K"] = K

    elif clf_name == "SVM":
        C = st.sidebar.slider("Regularization",0.01,10.0,step=0.01)
        kernel = st.sidebar.selectbox("Kernel",("linear", "poly", "rbf", "sigmoid", "precomputed"))
        params["C"] = C
        params["kernel"] = kernel

    elif clf_name == "Decision Trees":
        M = st.sidebar.slider("max_depth", 2, 20)
        C = st.sidebar.selectbox("Criterion", ("gini", "entropy"))
        SS = st.sidebar.slider("min_samples_split",1,10)
        params["M"] = M
        params["C"] = C
        params["SS"] = SS

    elif clf_name == "Random Forest":
        N = st.sidebar.slider("n_estimators",50,500,step=50,value=100)
        M = st.sidebar.slider("max_depth",2,20)
        C = st.sidebar.selectbox("Criterion",("gini","entropy"))
        params["N"] = N
        params["M"] = M
        params["C"] = C

    elif clf_name == "Gradient Boosting":
        N = st.sidebar.slider("n_estimators", 50, 500, step=50,value=100)
        LR = st.sidebar.slider("Learning Rate", 0.01, 0.5)
        L = st.sidebar.selectbox("Loss", ('deviance', 'exponential'))
        M = st.sidebar.slider("max_depth",2,20)
        params["N"] = N
        params["LR"] = LR
        params["L"] = L
        params["M"] = M

    elif clf_name == "XGBoost":
        N = st.sidebar.slider("n_estimators", 50, 500, step=50, value=50)
        LR = st.sidebar.slider("Learning Rate", 0.01, 0.5,value=0.1)
        O = st.sidebar.selectbox("Objective", ('binary:logistic','reg:logistic','reg:squarederror',"reg:gamma"))
        M = st.sidebar.slider("max_depth", 1, 20,value=6)
        G = st.sidebar.slider("Gamma",0,10,value=5)
        L = st.sidebar.slider("reg_lambda",1.0,5.0,step=0.1)
        A = st.sidebar.slider("reg_alpha",0.0,5.0,step=0.1)
        CS = st.sidebar.slider("colsample_bytree",0.5,1.0,step=0.1)
        params["N"] = N
        params["LR"] = LR
        params["O"] = O
        params["M"] = M
        params["G"] = G
        params["L"] = L
        params["A"] = A
        params["CS"] = CS
    
    elif clf_name == "CNN":
        LR = st.sidebar.slider("Learning Rate", 0.01, 0.5,value=0.1)
        E = st.sidebar.slider("Epochs", 1, 100, value=1)
        params["LR"] = LR
        params["E"] = E
        
    if clf_name != "CNN":
        RS=st.sidebar.slider("Random State",0,100)
        params["RS"] = RS
    return params

if train_inf == 'train':
    params = add_parameter_ui(classifier_name)
    flag = st.sidebar.button('Apply')
#print(params)
#print(flag)
	
def get_classifier(clf_name,params, data_type='None'):
    global clf
    if clf_name == "Logistic Regression":
        clf = LogisticRegression(C=params["R"],max_iter=params["MI"])

    elif clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])

    elif clf_name == "SVM":
        clf = SVC(kernel=params["kernel"],C=params["C"])

    elif clf_name == "Decision Trees":
        clf = DecisionTreeClassifier(max_depth=params["M"],criterion=params["C"]) #,min_impurity_split=params["SS"])

    elif clf_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=params["N"],max_depth=params["M"],criterion=params["C"])

    elif clf_name == "Gradient Boosting":
        clf = GradientBoostingClassifier(n_estimators=params["N"],learning_rate=params["LR"],loss=params["L"],max_depth=params["M"])

    elif clf_name == "XGBoost":
    	clf = XGBClassifier(use_label_encoder=False, objective = 'binary:logistic', nthread=4, seed=10)
    
    elif clf_name == 'CNN':
        if data_type == 'Image':
            clf = torchvision.models.resnet18(pretrained=True)
            clf.fc = torch.nn.Linear(in_features=512, out_features=classes)
        elif data_type == 'Audio':
            clf = CoughNet(len(hparams['features'])) #.to(device)
        #st.write(clf)
    return clf, clf_name

if train_inf == 'train':
    if dataset_name in ['Skin Cancer', 'Covid-19']:
        clf, clf_name = get_classifier(classifier_name,params, data_type)
    else:
        clf, clf_name = get_classifier(classifier_name,params)


	
#Build Model
def model():
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=65)

    #MinMax Scaling / Normalization of data
    Std_scaler = StandardScaler()
    X_train = Std_scaler.fit_transform(X_train)
    X_test = Std_scaler.transform(X_test)

    clf.fit(X_train,Y_train)
    Y_pred = clf.predict(X_test)
    acc=accuracy_score(Y_test,Y_pred)

    return Y_pred,Y_test
    
def train(epochs):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(clf.parameters(), lr=params['LR'])
    print('Starting training..')
    for e in range(0, params['E']):
        print('='*20)
        print(f'Starting epoch {e + 1}/{epochs}')
        print('='*20)

        train_loss = 0.
        val_loss = 0.

        clf.train() # set model to training phase

        for train_step, (images, labels) in enumerate(train_dataset):
            optimizer.zero_grad()
            print(images.shape)
            outputs = clf(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if train_step % 20 == 0:
                print('Evaluating at step', train_step)

                accuracy = 0

                clf.eval() # set model to eval phase

                for val_step, (images, labels) in enumerate(val_dataset):
                    print(images.shape)
                    outputs = clf(images)
                    loss = loss_fn(outputs, labels)
                    val_loss += loss.item()

                    _, preds = torch.max(outputs, 1)
                    accuracy += sum((preds == labels).numpy())

                val_loss /= (val_step + 1)
                accuracy = accuracy/len(val_dataset)
                print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')

                #show_preds()

                clf.train()

                # if accuracy >= 0.95:
                #     print('Performance condition satisfied, stopping..')
                #     return

        train_loss /= (train_step + 1)

        print(f'Training Loss: {train_loss:.4f}')
    st.write('Training complete..')

if train_inf == 'train':
    if flag :
        # The message and nested widget will remain on the page	
        if dataset_name == 'Covid-19' or dataset_name == "Skin Cancer":
            st.write("Training has started")
            train(params["E"])
        else:
            Y_pred,Y_test=model()

#Plot Output
def compute(Y_pred,Y_test):
    #Plot PCA
    pca=PCA(2)
    X_projected = pca.fit_transform(X)
    x1 = X_projected[:,0]
    x2 = X_projected[:,1]
    plt.figure(figsize=(16,8))
    plt.scatter(x1,x2,c=Y,alpha=0.8,cmap="viridis")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()
    st.pyplot()

    c1, c2 = st.beta_columns((4,3))
    #Output plot
    plt.figure(figsize=(12,6))
    plt.scatter(range(len(Y_pred)),Y_pred,color="yellow",lw=5,label="Predictions")
    plt.scatter(range(len(Y_test)),Y_test,color="red",label="Actual")
    plt.title("Prediction Values vs Real Values")
    plt.legend()
    plt.grid(True)
    c1.pyplot()

    #Confusion Matrix
    cm=confusion_matrix(Y_test,Y_pred)
    class_label = ["High-risk", "Low-risk"]
    df_cm = pd.DataFrame(cm, index=class_label,columns=class_label)
    plt.figure(figsize=(12, 7.5))
    sns.heatmap(df_cm,annot=True,cmap='Pastel1',linewidths=2,fmt='d')
    plt.title("Confusion Matrix",fontsize=15)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    c2.pyplot()

    #Calculate Metrics
    acc=accuracy_score(Y_test,Y_pred)
    mse=mean_squared_error(Y_test,Y_pred)
    precision, recall, fscore, train_support = score(Y_test, Y_pred, pos_label=1, average='binary')
    st.subheader("Metrics of the model: ")
    st.text('Precision: {} \nRecall: {} \nF1-Score: {} \nAccuracy: {} %\nMean Squared Error: {}'.format(
        round(precision, 3), round(recall, 3), round(fscore,3), round((acc*100),3), round((mse),3)))

if flag:
	st.markdown("<hr>",unsafe_allow_html=True)
	st.header(f"1) Model for Prediction of {dataset_name}")
	st.subheader(f"Classifier Used: {classifier_name}")
	#Execution Time
	end_time=time.time()
	st.info(f"Total execution time: {round((end_time - start_time),4)} seconds")

if flag:
	if dataset_name == "Heart Attack" or dataset_name == "Breast Cancer":
		compute(Y_pred,Y_test)


#Get user values
def user_inputs_ui(dataset_name,data):
    user_val = {}
    if dataset_name == "Breast Cancer":
        X = data.drop(["id","diagnosis"], axis=1)
        for col in X.columns:
            name=col
            col = st.number_input(col, abs(X[col].min()-round(X[col].std())), abs(X[col].max()+round(X[col].std())))
            user_val[name] = round((col),4)

    elif dataset_name == "Heart Attack":
        X = data.drop(["output"], axis=1)
        for col in X.columns:
            name=col
            col = st.number_input(col, abs(X[col].min()-round(X[col].std())), abs(X[col].max()+round(X[col].std())))
            user_val[name] = col

    return user_val

# if flag and dataset_name in ['Heart Attack',"Breast Cancer"]:
# 	#User values
# 	st.markdown("<hr>",unsafe_allow_html=True)
# 	st.header("2) User Values")
# 	with st.beta_expander("See more"):
# 	    st.markdown("""
# 	    In this section you can use your own values to predict the target variable. 
# 	    Input the required values below and you will get your status based on the values. <br>
# 	    <p style='color: red;'> 1 - High Risk </p> <p style='color: green;'> 0 - Low Risk </p>
# 	    """,unsafe_allow_html=True)

#@st.cache(suppress_st_warning=True)
def user_predict():
    global U_pred
    if dataset_name == "Breast Cancer":
        if clf_name == "XGBoost":
    	    X = data.drop(["id","diagnosis"], axis=1)
    	    features = np.array([[user_val[col] for col in X.columns]])
    	    U_pred = clf.predict(features)
        else:
            X = data.drop(["id","diagnosis"], axis=1)
            U_pred = clf.predict([[user_val[col] for col in X.columns]])

    elif dataset_name == "Heart Attack":
        if clf_name == "XGBoost":
    	    X = data.drop(["output"], axis=1)
    	    features = np.array([[user_val[col] for col in X.columns]])
    	    U_pred = clf.predict(features)
        else:
            X = data.drop(["output"], axis=1)
            U_pred = clf.predict([[user_val[col] for col in X.columns]])
    
    # predict = st.button('Predict')
    # if predict:
    st.subheader("Your Status: ")
    if U_pred == 0:
        st.write(U_pred[0], " - You are not at high risk :)")
    else:
        st.write(U_pred[0], " - You are at high risk :(")

def cnn_predict(img_file_buffer, dataset_name):
    preprocess = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),            
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if dataset_name == 'Skin Cancer':
        image = Image.open(img_file_buffer)
    elif dataset_name == 'Covid-19':
        image = Image.open(img_file_buffer).convert('RGB')
    # Apply the transformations
    input_tensor = preprocess(image)
    
    #print(input_tensor.shape)
    # Add an extra dimension to the tensor to create a batch
    input_batch = input_tensor.unsqueeze(0)

    # Convert the input batch to a PyTorch Variable
    input_batch = input_batch.to(torch.float32)

    clf.eval()

    #print(input_batch.shape)
    outputs = clf(input_batch)
    _, preds = torch.max(outputs, 1)
    #print(preds)
    # st.write(preds)
    # st.write(preds.data)

    if dataset_name == 'Skin Cancer':
        if preds.data == 0:
            st.write("You are having benign cancer")
        else:
            st.write("You are having malignant cancer")

    elif dataset_name == 'Covid-19':
        if preds.data == 0:
            st.write("You are out of risk")
        elif preds.data == 1:
            st.write("You are suffering from viral")
        elif preds.data == 2:
            st.write("You are suffering from covid")

def cnn_predict_audio(audio_file):
    loaded_checkpoint = torch.load('checkpoint.pth')

    scaler = loaded_checkpoint['scaler']
    encoder = loaded_checkpoint['encoder']

    clf.eval()
    df_features = pd.DataFrame(columns=hparams['features'])
    df_features = df_features.append(preproces(audio_file), ignore_index=True)
    X = np.array(df_features[hparams['features']], dtype=np.float32)
    X = torch.Tensor(scaler.transform(X))

    outputs = torch.softmax(clf(X), 1)
    predictions = torch.argmax(outputs.data, 1)
    
    #st.write(predictions)
    #st.write(f'model outputs {outputs[0].detach().numpy()} which predicts the class {encoder.classes_[predictions]}!')
    if encoder.classes_[predictions] == 'covid':
        st.write("You are tested covid positive")
    else:
        st.write("You are tested covid negative")
    #st.write("You are {encoder.classes_[predictions]})

## Dataset Type
if flag:
    if dataset_name in ['Heart Attack',"Breast Cancer"]:
        #Predict the status of user.
        st.markdown("<hr>",unsafe_allow_html=True)
        st.header("2) User Values")
        with st.beta_expander("See more"):
            st.markdown("""
                In this section you can use your own values to predict the target variable. 
                Input the required values below and you will get your status based on the values. <br>
                <p style='color: red;'> 1 - High Risk </p> <p style='color: green;'> 0 - Low Risk </p>
                """,unsafe_allow_html=True)
        user_val=user_inputs_ui(dataset_name,data)
        user_predict()  

if dataset_name == 'Skin Cancer':
    img_file_buffer = st.file_uploader("Choose an image", type=['png', 'jpg'])
    if img_file_buffer is not None:
        cnn_predict(img_file_buffer, dataset_name)
elif dataset_name == 'Covid-19':
    if data_type == "Image":
        img_file_buffer = st.file_uploader("Choose an image", type=['png', 'jpg'])
        if img_file_buffer is not None:
            cnn_predict(img_file_buffer, dataset_name)
    else:
        audio_file = st.file_uploader("Upload an audio file", type=["mp3","wav"])
        if audio_file is not None:
            cnn_predict_audio(audio_file)
#-------------------------------------------------------------------------END------------------------------------------------------------------------#
