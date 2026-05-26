#!/usr/bin/env python
# coding: utf-8

# In[2]:


import IPython.display as ipd
import librosa
import librosa.display
import pandas as pd
import os, time, warnings
import seaborn as sns
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv1D,
    MaxPooling1D,
    BatchNormalization,
    Dropout,
    Flatten,
    Conv2D,
    MaxPool2D,
)

warnings.filterwarnings("ignore")


# In[ ]:


import kagglehub

# Download latest version
path = kagglehub.dataset_download("sreyareddy15/esc10rearranged")

print("Path to dataset files:", path)


# In[ ]:


import os

print(os.listdir(path))


# In[ ]:


import os

print(os.listdir(os.path.join(path, "Data")))


# In[ ]:


DATA_PATH = os.path.join(path, "Data")


# Ten kod przechodzi przez folder z danymi audio, zbiera wszystkie pliki .wav, przypisuje im klasy (nazwy folderów), a potem tworzy z tego tabelę DataFrame

# In[ ]:


def get_ds_paths(input_dir) -> []:
    input_files = []
    input_classes = []
    for dirs, subdirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.wav'):
                input_files.append(file)
                input_classes.append(dirs.split('/')[-1])
    return np.array(input_files), np.array(input_classes)

files, classes = get_ds_paths(DATA_PATH)
data = { "file": files, "class": classes}
df = pd.DataFrame(data)
df


# Sprawdzamy rozłożenie klas w naszym zbiorze danych

# In[ ]:


x = df["class"].unique()
y = df["class"].value_counts(ascending=True)
ind = np.arange(len(y))
# plt.figure()
fig, ax = plt.subplots(figsize=(15, 5))
ax.barh(ind, y)
ax.set_yticks(ind)
ax.set_yticklabels(x)
ax.bar_label(ax.containers[0])
plt.gcf().set_dpi(500)
plt.title("Number of Audio Samples per Category")
plt.xlabel("Number of Samples")
plt.ylabel("Category")
plt.show()


# Poniższy kod:
# 
# 1. przechodzi przez wszystkie pliki audio,
# 2. wczytuje każdy .wav,
# 3. wyciąga cechy MFCC,
# 4. zamienia je na wektor liczb,
# 5. tworzy dataset gotowy do uczenia modelu ML.
# 
# 

# In[ ]:


extracted = []

for index_num, row in tqdm(df.iterrows()):
    file_name = os.path.join(os.path.abspath(DATA_PATH) + "/" + str(row["class"]) + "/" + str(row["file"]))
    final_class_labels = row["class"]
    # load the audio file
    audio, sample_rate = librosa.load(file_name, res_type="kaiser_fast")
    # extract the features
    feature = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=128)
    # feature scaling
    scaled_feature = np.mean(feature.T, axis=0)
    # store it in a list
    extracted.append([scaled_feature, final_class_labels])

extracted_df = pd.DataFrame(extracted, columns=["feature", "class"])
extracted_df.head()


# 

# In[ ]:


X = np.array(extracted_df["feature"].tolist())
y = np.array(extracted_df["class"].tolist())

# label encoding to get encoding
le = LabelEncoder()
# transform each category with it's respected label
Y = to_categorical(le.fit_transform(y))


# In[ ]:


# split the data to train and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# print the details
print("Number of training samples = ", X_train.shape[0])
print("Number of testing samples = ", X_test.shape[0])


# Tworzymy model sieci neuronowej

# In[ ]:


num_labels = Y.shape[1]
ANN_Model = Sequential()
ANN_Model.add(Dense(1000, activation="relu", input_shape=(128,)))
ANN_Model.add(Dense(750, activation="relu"))
ANN_Model.add(Dense(500, activation="relu"))
ANN_Model.add(Dropout(0.3))
ANN_Model.add(Dense(250, activation="relu"))
ANN_Model.add(Dense(100, activation="relu"))
ANN_Model.add(Dense(50, activation="relu"))
ANN_Model.add(Dense(num_labels, activation="softmax"))

ANN_Model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)

ANN_Model.summary()


# Poniższy kod:
# 1. trenuje model neuronowy,
# 2. zapisuje wytrenowany model,
# 3. zapisuje historię treningu,

# In[ ]:


num_epochs = 150
num_batch_size = 32

t0 = time.time()
ANN_Results = ANN_Model.fit(
    X_train,
    y_train,
    batch_size=num_batch_size,
    epochs=num_epochs,
    validation_data=(X_test, y_test),
)

ANN_Model.save("Model1.h5")
print("ANN Model Saved")
train_hist_m1 = pd.DataFrame(ANN_Results.history)
train_m1 = round(time.time() - t0, 3)


# In[ ]:


plt.figure(figsize=(8, 5))
plt.plot(train_hist_m1[["loss", "val_loss"]])
plt.legend(["Loss", "Validation Loss"])
plt.title("Loss Per Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()


# In[ ]:


plt.figure(figsize=(8, 5))
plt.plot(train_hist_m1[["accuracy", "val_accuracy"]])
plt.legend(["Accuracy", "Validation Accuracy"])
plt.title("Accuracy Per Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()


# Ten kod tworzy system do porównywania modeli ML/AI — zapisuje accuracy, czas treningu i czas predykcji do jednej tabeli

# In[ ]:


log_cols = ["model", "accuracy", "train_time", "pred_time"]
log = pd.DataFrame(columns=log_cols)

def Log_Model(model, model_name, x, y, t):
    global log
    acc = model.evaluate(x, y, verbose=0)
    t0 = time.time()
    y_pred = model.predict(x, verbose=0)
    pred = round(time.time() - t0, 3)
    log_entry = pd.DataFrame([[model_name, acc[1] * 100, t, pred]], columns=log_cols)
    log = pd.concat([log, log_entry], ignore_index=True)


# In[ ]:


Log_Model(ANN_Model, "ANN", X_test, y_test, train_m1)
log


# In[ ]:


def ANN_Prediction(file_name):
    # load the audio file
    audio_data, sample_rate = librosa.load(file_name, res_type="kaiser_fast")
    # get the feature
    feature = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=128)
    # scale the features
    feature_scaled = np.mean(feature.T, axis=0)
    # array of features
    prediction_feature = np.array([feature_scaled])
    # get the id of label using argmax
    predicted_vector = np.argmax(ANN_Model.predict(prediction_feature), axis=-1)
    # get the class label from class id
    predicted_class = le.inverse_transform(predicted_vector)
    # display the result
    print("ANN has predicted the class as  --> ", predicted_class[0])

# File name
file_name = DATA_PATH + "/rooster/2-96460-A-1.wav"
# get the output
ANN_Prediction(file_name)
# play the file
ipd.Audio(file_name)


# In[ ]:


xTrainval, xTest, yTrainval, yTest = train_test_split(
    X, Y, test_size=0.1, stratify=y, random_state=387
)
xTrain, xvalid, yTrain, yvalid = train_test_split(
    xTrainval, yTrainval, test_size=0.2, stratify=yTrainval, random_state=387
)
print("\nNumber of samples for Train set :", xTrain.shape[0])
print("Number of samples for Validation set :", xvalid.shape[0])
print("Number of samples for Test set :", xTest.shape[0])

xTrain = np.expand_dims(xTrain, axis=2)
xvalid = np.expand_dims(xvalid, axis=2)

print("Shape of X Train", xTrain.shape)
print("Shape of X Test", xTest.shape)


# Tworzymy model sieci CNN

# In[ ]:


CNN1D_Model = Sequential()
CNN1D_Model.add(
    Conv1D(
        256,
        5,
        strides=1,
        padding="same",
        activation="relu",
        input_shape=(xTrain.shape[1], 1),
    )
)
CNN1D_Model.add(BatchNormalization())
CNN1D_Model.add(MaxPooling1D(3, strides=2, padding="same"))
CNN1D_Model.add(Conv1D(128, 5, strides=1, padding="same", activation="relu"))
CNN1D_Model.add(Dropout(0.3))
CNN1D_Model.add(MaxPooling1D(3, strides=2, padding="same"))
CNN1D_Model.add(Conv1D(128, 5, strides=1, padding="same", activation="relu"))
CNN1D_Model.add(Dropout(0.2))
CNN1D_Model.add(MaxPooling1D(3, strides=2, padding="same"))
CNN1D_Model.add(Conv1D(64, 5, strides=1, padding="same", activation="relu"))
CNN1D_Model.add(Dropout(0.3))
CNN1D_Model.add(MaxPooling1D(3, strides=2, padding="same"))
CNN1D_Model.add(Flatten())
CNN1D_Model.add(Dense(units=1024, activation="relu"))
CNN1D_Model.add(Dropout(0.2))
CNN1D_Model.add(Dense(units=num_labels, activation="softmax"))
CNN1D_Model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
CNN1D_Model.summary()


# In[ ]:


t0 = time.time()

CNN1D_Results = CNN1D_Model.fit(
    xTrain, yTrain, batch_size=num_batch_size, epochs=num_epochs, validation_data=(xvalid, yvalid)
)

CNN1D_Model.save("Model2.h5")
print("CNN1D Model Saved")
train_hist_m2 = pd.DataFrame(CNN1D_Results.history)
train_m2 = round(time.time() - t0, 3)


# In[ ]:


plt.figure(figsize=(8, 5))
plt.plot(train_hist_m2[["loss", "val_loss"]])
plt.legend(["Loss", "Validation Loss"])
plt.title("Loss Per Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()


# In[ ]:


plt.figure(figsize=(8, 5))
plt.plot(train_hist_m2[["accuracy", "val_accuracy"]])
plt.legend(["Accuracy", "Validation Accuracy"])
plt.title("Accuracy Per Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()


# In[ ]:


Log_Model(CNN1D_Model, "CNN1D", xvalid, yvalid, train_m2)
log


# In[ ]:


def CNN1D_Prediction(file_name):
    # load the audio file
    audio_data, sample_rate = librosa.load(file_name, res_type="kaiser_fast")
    # get the feature
    feature = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=128)
    # scale the features
    feature_scaled = np.mean(feature.T, axis=0)
    # array of features
    prediction_feature = np.array([feature_scaled])
    # expand dims
    final_prediction_feature = np.expand_dims(prediction_feature, axis=2)
    # get the id of label using argmax
    predicted_vector = np.argmax(CNN1D_Model.predict(final_prediction_feature), axis=-1)
    # get the class label from class id
    predicted_class = le.inverse_transform(predicted_vector)
    # display the result
    print("CNN1D has predicted the class as  --> ", predicted_class[0])

# File name
file_name = DATA_PATH + "/rain/4-161127-A-10.wav"
# get the output
CNN1D_Prediction(file_name)
# play the file
ipd.Audio(file_name)


# In[ ]:


xtrain = xTrain.reshape(xTrain.shape[0], 16, 8, 1)
xtest = xTest.reshape(xTest.shape[0], 16, 8, 1)

print("The Shape of X Train", xtrain.shape)
print("The Shape of Y Train", yTrain.shape)
print("The Shape of X Test", xtest.shape)
print("The Shape of Y Test", yTest.shape)


# In[ ]:


CNN2D_Model = Sequential()
CNN2D_Model.add(
    Conv2D(64, (3, 3), padding="same", activation="tanh", input_shape=(16, 8, 1))
)
CNN2D_Model.add(MaxPool2D(pool_size=(2, 2)))
CNN2D_Model.add(Conv2D(128, (3, 3), padding="same", activation="tanh"))
CNN2D_Model.add(MaxPool2D(pool_size=(2, 2)))
CNN2D_Model.add(Dropout(0.2))
CNN2D_Model.add(Flatten())
CNN2D_Model.add(Dense(1024, activation="tanh"))
CNN2D_Model.add(Dense(128, activation="relu"))
CNN2D_Model.add(Dense(num_labels, activation="softmax"))
CNN2D_Model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
CNN2D_Model.summary()


# In[ ]:


t0 = time.time()

CNN2D_Results = CNN2D_Model.fit(
    xtrain, yTrain, epochs=num_epochs, batch_size=num_batch_size, validation_data=(xtest, yTest)
)

CNN2D_Model.save("Model3.h5")
print("CNN2D Model Saved")
train_hist_m3 = pd.DataFrame(CNN2D_Results.history)
train_m3 = round(time.time() - t0, 3)


# In[ ]:


plt.figure(figsize=(8, 5))
plt.plot(train_hist_m3[["loss", "val_loss"]])
plt.legend(["Loss", "Validation Loss"])
plt.title("Loss Per Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()


# In[ ]:


plt.figure(figsize=(8, 5))
plt.plot(train_hist_m3[["accuracy", "val_accuracy"]])
plt.legend(["Accuracy", "Validation Accuracy"])
plt.title("Accuracy Per Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()


# In[ ]:


Log_Model(CNN2D_Model, "CNN2D", xtest, yTest, train_m3)
log


# In[ ]:


# function to predict the feature
def CNN2D_Prediction(file_name):
    # load the audio file
    audio_data, sample_rate = librosa.load(file_name, res_type="kaiser_fast")
    # get the feature
    feature = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=128)
    # scale the features
    feature_scaled = np.mean(feature.T, axis=0)
    # array of features
    prediction_feature = np.array([feature_scaled])
    # reshaping the features
    final_prediction_feature = prediction_feature.reshape(
        prediction_feature.shape[0], 16, 8, 1
    )
    # get the id of label using argmax
    predicted_vector = np.argmax(CNN2D_Model.predict(final_prediction_feature), axis=-1)
    # get the class label from class id
    predicted_class = le.inverse_transform(predicted_vector)
    # display the result
    print("CNN2D has predicted the class as  --> ", predicted_class[0])

# File name
file_name = DATA_PATH + "/dog/2-114587-A-0.wav"
# get the output
CNN2D_Prediction(file_name)
# play the file
ipd.Audio(file_name)


# In[ ]:


plt.rcParams["figure.figsize"] = (17, 2)
plt.rcParams["figure.dpi"] = 550

def plot_log(what):
    ax = sns.barplot(x=what, y="model", data=log)
    ax.bar_label(ax.containers[0])
    plt.xlabel("Accuracy")
    plt.ylabel("Model")
    plt.title("Model Accuracy")
    return plt.show()

plot_log("accuracy")
plot_log("train_time")
plot_log("pred_time")

