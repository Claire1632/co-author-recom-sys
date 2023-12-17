import pandas as pd
import sklearn, os, json, numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
import sqlite3, pickle
import collections




basedir = os.path.dirname((os.path.dirname(__file__)))
results_path = os.path.join(basedir, 'D:/MULTIMEDIA/MACHINE_LEARNING_THAY_QUANG/DOCUMENTS/NC1/Compression/Dongtacgia/Project3/Project3_New/Results')
models_path = os.path.join(basedir, 'D:/MULTIMEDIA/MACHINE_LEARNING_THAY_QUANG/DOCUMENTS/NC1/Compression/Dongtacgia/Project3/Project3_New/Models')
db_path = os.path.join(os.path.dirname(basedir), 'D:/MULTIMEDIA/MACHINE_LEARNING_THAY_QUANG/DOCUMENTS/NC1/Compression/Dongtacgia/Project3/Project3_New/Data_Project3')


def get_test_authors(data_name, test_percent):
    data_path = results_path + "/" + data_name
    data = pd.read_csv(data_path)
    data = data.drop_duplicates(subset=['CommonNeighbor', 'AdamicAdar', 'JaccardCoefficient', 'PreferentialAttachment', 'ResourceAllocation', 'ShortestPath' ,'CommonCountry', 'Label'])
    X = data.drop(columns=['id_author_1', 'id_author_2', 'Label'])
    y = data['Label']
    # print(X.shape)
    # print("Tỉ lệ nhãn -1-1")
    # print()
    # print(np.sum(y==-1), end='--')
    # print(np.sum(y==1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_percent/100), random_state=608, shuffle=True)
    
    list_test_authors = set()
    list_CN = list(X_test['CommonNeighbor'])
    list_AA = list(X_test['AdamicAdar'])
    list_JC = list(X_test['JaccardCoefficient'])
    list_PA = list(X_test['PreferentialAttachment'])
    list_RA = list(X_test['ResourceAllocation'])
    list_SP = list(X_test['ShortestPath'])
    list_CC = list(X_test['CommonCountry'])

    for i in range(len(list_CN)):
        tmp_df = X_test[(X_test['iCommonNeighbor'] == list_CN[i]) &
                        (X_test['AdamicAdar'] == list_AA[i]) &
                        (X_test['JaccardCoefficient'] == list_JC[i]) &
                        (X_test['PreferentialAttachment'] == list_PA[i]) &
                        (X_test['ResourceAllocation'] == list_RA[i]) &
                        (X_test['ShortestPath'] == list_SP[i]) &
                        (X_test['CommonCountry'] == list_CC[i])
        ]
        for id1 in list(tmp_df['id_author_1']): #ban dau:tmp
            list_test_authors.add(id1)
        for id2 in list(tmp_df['id_author_2']): #ban dau:tmp
            list_test_authors.add(id2)
         
    list_id_names = []
    with sqlite3.connect(db_path + '/db.sqlite3') as conn:
        cur = conn.cursor()
        query = ("select id, first_name, last_name from collab_author \
                    where id in ({seq})"
                .format(seq=','.join(['?']*len(list_test_authors))))
        cur.execute(query, list_test_authors)
        result = cur.fetchall()
        for id, first_name, last_name in result:
            list_id_names.append((id, first_name + " " + last_name))
        return list_id_names

def train(data_name, test_percent):
    test_percent = int(test_percent)
    print(test_percent)
    data_path = results_path + "/" + data_name
    print(data_path)
    data = pd.read_csv(data_path)
    print(data.shape)
    # data = data.drop_duplicates(subset=['CommonNeighbor', 'AdamicAdar', 'JaccardCoefficient', 'PreferentialAttachment', 'ResourceAllocation', 'ShortestPath' ,'CommonCountry', 'Label'])
    # print(data.shape)
    X = data.drop(columns=['id_author_1', 'id_author_2', 'Label'])
    y = data['Label']
    print(X.shape)
    print("Tỉ lệ nhãn -1-1")
    print()
    print(np.sum(y==-1), end='--')
    print(np.sum(y==1))

    scaler = MinMaxScaler().fit(X)
    X = scaler.transform(X)
    
    # Note: this is only for the purpose of demo. 
    # To be precise, train set and test set should be created by splitting the whole papers into two consecutive sets
    # For example: train set consists of papers from 2000-2008
    # test set consists of papers from 2009-2017
    # Then code should be like this
    # data_train = pd.read_csv(results_path + "/" + "train.csv")
    # data_test = pd.read_csv(results_path + "/" + "test.csv")

    # data_train = data_train.drop_duplicates(subset=['CommonNeighbor', 'AdamicAdar', 'JaccardCoefficient', 'PreferentialAttachment', 'ResourceAllocation', 'ShortestPath' ,'CommonCountry', 'Label'])
    # data_test = data_test.drop_duplicates(subset=['CommonNeighbor', 'AdamicAdar', 'JaccardCoefficient', 'PreferentialAttachment', 'ResourceAllocation', 'ShortestPath' ,'CommonCountry', 'Label'])

    # X_train = data_train.drop(columns=['id_author_1', 'id_author_2', 'Label', 'CommonNeighbor', 'AdamicAdar', 'JaccardCoefficient', 'PreferentialAttachment', 'ResourceAllocation', 'ShortestPath'])
    # y_train = data_train['Label']

    # X_test = data_test.drop(columns=['id_author_1', 'id_author_2', 'Label', 'CommonNeighbor', 'AdamicAdar', 'JaccardCoefficient', 'PreferentialAttachment', 'ResourceAllocation', 'ShortestPath'])
    # y_test = data_test['Label'] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_percent/100), random_state=608, shuffle=True)

    # print(X_train[:10])

    model = svm.SVC(kernel='rbf', max_iter=5000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Precision:",metrics.precision_score(y_test, y_pred))
    print("Recall:",metrics.recall_score(y_test, y_pred))
    print("F1 score:",metrics.f1_score(y_test, y_pred))
    print("Roc_auc_score score:",metrics.roc_auc_score(y_test, y_pred))
    
    result = {}
    result['Precision'] = metrics.precision_score(y_test, y_pred)
    result['Recall'] = metrics.recall_score(y_test, y_pred)
    result['f1_score'] = metrics.f1_score(y_test, y_pred)
    result['Roc_auc'] = metrics.roc_auc_score(y_test, y_pred)

    tmp = data_name.split('_')
    
    tmp[0] = "Model"
    model_name = "_".join(tmp)[:-4] + ".pkl"
    with open(models_path + '/' + model_name, 'wb') as file:
        pickle.dump(model, file)

    tmp[0] = "Scaler"
    scaler_name = "_".join(tmp)[:-4] + ".pkl"
    with open(models_path + '/' + scaler_name, 'wb') as file:
        pickle.dump(scaler, file)


    return json.dumps({"results": result, "model_name": model_name})

