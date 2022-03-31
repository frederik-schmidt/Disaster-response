# Disaster Response Pipeline Project

## 1. Project Overview

The motivation behind this project is to classify disaster messages into categories. Through a web app, users can input a new disaster message and get classification results in several categories. With this classification, help can be organized in an efficient way.

The data was provided by [Figure Eight](https://www.figure-eight.com/) (now [Appen](https://appen.com/)) and [Udacity](https://www.udacity.com/) and originally comes from real messages that were sent during disaster events.

More detailed:
- The data basis contains 26,248 unique messages, such as `Please, we need tents and water. We are in Silo, Thank you!`.
- Every message contains True/Falue values for 36 categories, such as `aid_related`, `weather_related`, `floods`, `earthquake`.
- A Random Forest classification model was applied for predicting the categories of new messages.
- The classification model achieves an accuracy of ~0.945.

## 2. Project Steps

The pipeline contains three steps:

### 2.1 ETL Pipeline

The code for the ETL Pipeline is present in `../data/process_data.py`. In addition, there is a Jupyter Notebook with these steps in `..\data\process_data.ipynb`.

1. Extract: Reads the `disaster_messages.csv` and `disaster_categories.csv` data
2. Transform: Merges the data and clean the merged data by tokenizing text data etc.
3. Load: Writes the clean data into a SQLite database

Run the ETL Pipeline:

```
> cd data/
> python process_data.py disaster_categories.csv disaster_messages.csv ../data/DisasterResponse.db
```

Output:

![alt text](/img/etl_pipeline.png "ETL Pipeline")

### 2.2 Machine Learning Pipeline

The code for the Machine Learning Pipeline is present in `../models/train_classifier.py`. In addition, there is a Jupyter Notebook with these steps in `..\model\train_classifier.ipynb`.

1. Load: Loads data from SQLite database and split into training and test set
2. Build: Builds a model using sklearn's Pipeline
3. Train: Trains the model using training data
4. Evaluate: Evaluates the model using test data
5. Save: Saves the trained model to a pickle file

Run the Machine Learning Pipeline (this takes up to 30 min):

```
> cd ../models/
> python train_classifier.py ../data/DisasterResponse.db disaster_response_model.pkl
```

Output:

![alt text](/img/ml_pipeline.png "Machine Learning Pipeline")

### 2.3 Web App

The code for the Web App is present in `../app/run.py`. 

Launch the Web App:

```
> cd ../app/
> python run.py
```

After launching, open the link `http://127.0.0.1:3000` in your browser to open the web app. Here, new messages can be classified, as shown in the screenshot below.

![alt text](/img/web_app.png "Disaster Response Project")

## 3. Installation Guide

### 3.1 Clone the repository

`> git clone https://github.com/frederik-schmidt/Disaster-response.git`

### 3.2 Create a virtual environment

It is highly recommended to use virtual environments to install packages.

`> conda create -n disaster_response python=3.8 jupyterlab`
(where `disaster_response` is the environment name of your choice)

### 3.3 Activate the virtual environment

`> conda activate disaster_response`
(or whatever environment name you used)

### 3.4 Install the required packages

```
> cd Disaster-response
> pip install -r requirements.txt
```

When the packages have been installed, everything is set up and ready to run the project steps described in section 2.

## 4. File Structure

```
.
├── app
│   ├── run.py
│   └── templates
│       ├── go.html
│       └── master.html
├── data
│   ├── disaster_categories.csv
│   ├── disaster_messages.csv
│   └── process_data.py
├── img
│   ├── etl_pipeline.png
│   ├── ml_pipeline.png
│   └── web_app.png
├── models
│   └── train_classifier.py
├── README.md
└── requirements.txt
```

## 5. Requirements

The project uses Python 3.8 and additional libraries:

- flask
- json
- nltk
- numpy
- os
- pandas
- pickle
- plotly
- re
- sklearn
- sqlalchemy
- sys
- warnings

## 6. Links

- [Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025)
