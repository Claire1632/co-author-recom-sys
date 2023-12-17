import pandas as pd
import sklearn, os, json, numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
import sqlite3, pickle
##################
import numpy as np
# from data import Vertebral_column
# from data import Co_Author
# from Processing_Data import Abanole
# from Processing_Data import Ecoli
# from Processing_Data import Ecloli1
# from Processing_Data import Ecoli3
# from Processing_Data import Glass1
# from Processing_Data import Glass4
# from Processing_Data import Haberman
# from Processing_Data import Waveform
# from Processing_Data import New_thyroid2
# from Processing_Data import Page_blocks
# from Processing_Data import Pima_Indians_Diabetes
# from Processing_Data import Satimage
# from Processing_Data import Transfusion
# from Processing_Data import Yeast
# from Processing_Data import Haberman_All
# from Processing_Data import Transfusion_All
# from Processing_Data import PimaIndians_All
# from data import indian_liver_patient
# #from data import spect_heart
# from wsvm.application import Wsvm
# from svm.application import Svm
from sklearn.svm import SVC
#from sklearn.metrics import f1_score
from sklearn.metrics  import classification_report,precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score, confusion_matrix,roc_auc_score,f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import _safe_indexing
from sklearn import metrics
import math
from datetime import datetime
# from fuzzy.weight import fuzzy
# from sklearn.model_selection import StratifiedKFold
# from sklearn.preprocessing import StandardScaler
# from Processing_Data import Ecoli_Kfold
# from Processing_Data import Haberman_KFold
# from Processing_Data import Transfution_Kfold
# import csv
# from Processing_Data import Satimage_KFold
# from Processing_Data import Yeast_KFold
# from Processing_Data import Co_Author

def svm_lib(X_train, y_train,X_test):
    svc=SVC(probability=True, kernel='linear')
    model = svc.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

# def wsvm(C,models_path,X_train, y_train,X_test,time,namefunc,namemethod,distribution_weight=None):
#     model = Wsvm(C,distribution_weight)
#     model.fit(X_train, y_train)
#     model_name = "Model_"+namemethod+"_"+namefunc+"_"+time+".pkl"
#     with open(models_path + '/' + model_name, 'wb') as file:
#         pickle.dump(model, file)
#     test_pred = model.predict(X_test)
#     return test_pred

# def svm(C,X_train, y_train,X_test):
#     model = Svm(C)
#     model.fit(X_train, y_train)
#     test_pred = model.predict(X_test)
#     return test_pred

def is_tomek(X,y, class_type):
    print(y)
    print(type(y))
    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(X)
    nn_index = nn.kneighbors(X, return_distance=False)[:, 1]
    links = np.zeros(len(y), dtype=bool)
    # find which class to not consider
    class_excluded = [c for c in np.unique(y) if c not in class_type]
    X_dangxet = []
    X_tl = []
    # there is a Tomek link between two samples if they are both nearest
    # neighbors of each others.
    for index_sample, target_sample in enumerate(y):
        if target_sample in class_excluded:
            continue
        if y[nn_index[index_sample]] != target_sample:
            if nn_index[nn_index[index_sample]] == index_sample:
                X_tl.append(index_sample)
                X_dangxet.append(nn_index[index_sample])
                links[index_sample] = True

    return links,X_dangxet,X_tl

def Gmean(y_test,y_pred):
    cm_WSVM = metrics.confusion_matrix(y_test, y_pred)
    sensitivity = cm_WSVM[1,1]/(cm_WSVM[1,0]+cm_WSVM[1,1])
    specificity = cm_WSVM[0,0]/(cm_WSVM[0,0]+cm_WSVM[0,1])
    gmean = math.sqrt(sensitivity*specificity)
    return specificity,sensitivity,gmean

def metr(X_train,y_test,test_pred,se,sp,gmean):
    #se,sp,gmean = Gmean(y_test,test_pred)
    print("So luong samples: ",len(X_train))
    print("\n",classification_report(y_test, test_pred))
    print("SP      : ",sp)
    print("SE      : ",se)
    print("Gmean   : ",gmean)
    print("F1 Score: ",f1_score(y_test, test_pred))
    print("Accuracy: ",accuracy_score(y_test,test_pred))
    print("AUC     : ",roc_auc_score(y_test, test_pred))
    print("Ma tran nham lan: \n",confusion_matrix(y_test, test_pred))

def metr_text(f,X_train,y_test,test_pred,sp,se,gmean):
    #se,sp,gmean = Gmean(y_test,test_pred)
    f.write(f"\n\nSo luong samples Tong: {len(X_train)+len(y_test)}")
    f.write(f"\n\nSo luong samples training: {len(X_train)}")
    f.write(f"\nSo luong samples testing: {len(y_test)}\n")
    f.write("\n"+str(classification_report(y_test, test_pred)))
    f.write(f"\nSP      : {sp:0.4f}")
    f.write(f"\nSE      : {se:0.4f}")
    f.write(f"\nGmean   : {gmean:0.4f}")
    f.write(f"\nF1 Score: {f1_score(y_test, test_pred):0.4f}")
    f.write(f"\nAccuracy: {accuracy_score(y_test,test_pred):0.4f}")
    f.write(f"\nAUC     : {roc_auc_score(y_test, test_pred):0.4f}")
    f.write("\n\nMa tran nham lan: \n"+str(confusion_matrix(y_test, test_pred)))

# def compute_weight(X, y,name_method ="actual_hyper_lin", name_function = "exp", beta = None,C = None, gamma = None, u = None, sigma = None):
#     method = fuzzy.method()
#     function = fuzzy.function()
#     pos_index = np.where(y == 1)[0]
#     neg_index = np.where(y == -1)[0]
#     try:
#         if name_method == "own_class_center": 
#             d = method.own_class_center(X, y)
#         elif name_method == "estimated_hyper_lin": # actual_hyper_lin, own_class_center
#             d = method.estimated_hyper_lin(X, y)
#         elif name_method == "own_class_center_opposite":
#             d = method.own_class_center_opposite(X, y)
#         elif name_method == 'actual_hyper_lin':
#             d = method.actual_hyper_lin(X, y,C = C, gamma = gamma)
#         elif name_method == 'own_class_center_divided':
#             d = method.own_class_center_divided(X, y)
#         elif name_method == "distance_center_own_opposite_tam":
#             d_own, d_opp, d_tam = method.distance_center_own_opposite_tam(X,y)
#         else:
#             print('dont exist method')
        
#         if name_function == "lin":
#             W = function.lin(d)
#         elif name_function == "exp":
#             W = function.exp(d, beta)
#         elif name_function == "lin_center_own":
#             W = function.lin_center_own(d, pos_index,neg_index)
#         elif name_function == 'gau':
#             W = function.gau(d, u, sigma)
#         elif name_function == "func_own_opp_new":
#             W = function.func_own_opp_new(d_own,d_opp,pos_index,neg_index,d_tam)
#     except Exception as e:
#         print('dont exist function')
#         print(e)
#     pos_index = np.where(y == 1)[0]
#     neg_index = np.where(y == -1)[0]
#     r_pos = 1
#     r_neg = len(pos_index)/len(neg_index)
#     m = []
#     W = np.array(W)
#     m = W[pos_index]*r_pos
#     m = np.append(m, W[neg_index]*r_neg)
#     return m

# def fuzzy_weight(f,beta_center, beta_estimate, beta_actual,X_train, y_train,namemethod,namefunction):
#     if namemethod =="own_class_center_opposite" and namefunction == "exp":
#         distribution_weight = compute_weight(X_train, y_train,name_method = namemethod,name_function = namefunction,beta = beta_center)
#         f.write(f"\n\t Beta 'own_class_center_opposite' with exp = {beta_center}\n")
#     elif namemethod =="own_class_center" and namefunction == "exp":
#         distribution_weight = compute_weight(X_train, y_train,name_method = namemethod,name_function = namefunction,beta = beta_estimate)
#         f.write(f"\n\t Beta 'own_class_center' with exp = {beta_center}\n")
#     elif namemethod =="own_class_center_divided" and namefunction == "exp":
#         distribution_weight = compute_weight(X_train, y_train,name_method = namemethod,name_function = namefunction,beta = beta_estimate)
#         f.write(f"\n\t Beta 'own_class_center_divided' with exp = {beta_center}\n")
#     elif namemethod =="estimated_hyper_lin" and namefunction == "exp":
#         distribution_weight = compute_weight(X_train, y_train,name_method = namemethod,name_function = namefunction,beta = beta_estimate)
#         f.write(f"\n\t Beta 'estimated_hyper_lin' with exp = {beta_estimate}\n")
#     elif namemethod =="actual_hyper_lin" and namefunction == "exp":
#         distribution_weight = compute_weight(X_train, y_train,name_method = namemethod,name_function = namefunction,beta = beta_actual)
#         f.write(f"\n\t Beta 'actual_hyper_lin' with exp = {beta_actual}\n")
#     else:   
#         distribution_weight = compute_weight(X_train, y_train,name_method = namemethod,name_function = namefunction)
#     return distribution_weight

# def data_tomelinks(f,C,weight,X_test,y_test,X_train,y_train,n_neighbors,clf=None,namemethod=None,namefunction=None):
#     links,ind_posX,ind_negX = is_tomek(X_train,y_train,class_type=[-1.0])
#     #print(len(ind_posX))
#     new_W = weight
#     pos_index = np.where(y_train == 1)[0]
#     neg_index = np.where(y_train == -1)[0]
#     clf = Wsvm(C,new_W)
#     clf.fit(X_train, y_train)
#     y_predict = clf.predict(X_test)
#     specificity,sensitivity,gmean = Gmean(y_test,y_predict)
#     nn2 = NearestNeighbors(n_neighbors=n_neighbors)
#     nn2.fit(X_train)
#     y_nn = []
#     #
#     neg_pred = clf.predict(X_train[neg_index])
#     idx_neg_wrong = np.where(neg_pred != -1.0)
#     new_W[idx_neg_wrong] =  new_W[idx_neg_wrong]/(2) # giam manh

#     #
#     ind_nn = []
#     for ind,i in enumerate(ind_posX): 
#         y_pred = clf.predict([X_train[i]])#
#         if y_pred == -1.0:
#             ind_nn.append(ind)                          
#             knn_X = (nn2.kneighbors([X_train[i]])[1]).tolist() 
#             for j in knn_X[0]:
#                 y_nn.append(y_train[j])    # gom nhãn láng giềng của X_train[i] bị dự đoán sai vào y_nn
#         else:
#             new_W[ind_negX[ind]] = new_W[ind_negX[ind]]/(1.2)
#             new_W[i] = new_W[i]*(1.2**10)
#     ind_nn = np.array(ind_nn)
#     #print(ind_nn)
#     y_nn = np.array(y_nn)
#     if len(y_nn)>0:
#         y_nn = np.array_split(y_nn, len(y_nn)/n_neighbors) 
#     #print(len(y_nn))
#     for ind,i in enumerate(range(0,len(y_nn))):   #
#         if 1 not in y_nn[i][1:]:      # Nếu không có nhãn 1 xung quanh X_train[i] bị dự đoán sai => xóa X_train[i]
#             new_W[ind_posX[ind_nn[ind]]] = new_W[ind_posX[ind_nn[ind]]]/(2)
#             #print(ind_posX[ind_nn[ind]])
#         else:
#             # print("Old Neg: ",new_W[ind_negX[ind_nn[ind]]])
#             # print("Old Pos: ",new_W[ind_posX[ind_nn[ind]]])
#             new_W[ind_negX[ind_nn[ind]]] = new_W[ind_negX[ind_nn[ind]]]/(1.2)
#             new_W[ind_posX[ind_nn[ind]]] = new_W[ind_posX[ind_nn[ind]]]*(1.2)
#             # print("New Neg: ",new_W[ind_negX[ind_nn[ind]]])
#             # print("New Pos: ",new_W[ind_posX[ind_nn[ind]]])

#     return new_W,gmean,sensitivity

# def lfb(f,C,weight,namemethod,namefunction,T,X_test,y_test,X_train,y_train,n_neighbors,thamso1,thamso2): #loop find the best
#     gmax = 0
#     tmax = 0
#     semax = 0
#     t_semax = 0
#     for i in range(0,1):
#         f.write(f"\t\t Vong Lap thu: T = {i+1}")
#         f.write(f"\n===================================================================================================================")
#         f.write(f"\n\n\tFuzzy SVM name_method = '{namemethod}',name_function = '{namefunction}' starting...\n")
#         weight, gmeanfirst, SEfirst = data_tomelinks(f,C,weight,X_test,y_test,X_train,y_train,n_neighbors,clf=None,namemethod=namemethod,namefunction=namefunction)
#         #distribution_weight = fuzzy_weight(f,beta_center, beta_estimate, beta_actual,X_train, y_train,namemethod,namefunction)
#         clf = Wsvm(C,weight)
#         #print(X_train)
#         #print(y_train)
#         clf.fit(X_train, y_train)
#         #print(clf.predict([X_test_val[5]]))
#         pred2 = clf.predict(X_test)    #X_val
#         sp,se,gmean = Gmean(y_test,pred2) #y_val
#         pred_all = clf.predict(X_test)    #X_val
#         sp_all,se_all,gmean_all = Gmean(y_test,pred_all)

#         if(gmeanfirst > gmax):
#             gmax = gmeanfirst
#             tmax = i+100
#         if (gmean_all > gmax):
#             gmax = gmean_all
#             tmax = i
#         if (SEfirst > semax):
#             semax = SEfirst
#             t_semax = i+100
#         if (se_all > semax):
#             semax = se_all
#             t_semax = i   
#         #metr(X_train,y_test_val,pred2,sp,se,gmean)
#         f.write(f"\n\n\t****** Danh gia tren tap Test:\n")
#         metr_text(f,X_train,y_test,pred_all,sp_all,se_all,gmean_all)
#         # if ((gmeanfirst - gmean) <= thamso1) or (gmeanfirst > thamso2):
#         #     f.write("\n_____Gmean_ERROR!!!____\n")
#         #     print("\n_____Gmean_ERROR!!!____\n")
#         #     return X_train, y_train
#         # else:
#         #     gmean = gmeanfirst
#         f.write(f"\n===================================================================================================================\n")
#     f.write(f"\nFuzzy SVM name_method = '{namemethod}',name_function = '{namefunction}'")
#     f.write(f"\n*** T = {tmax}; K = {n_neighbors}; GmeanMax = {gmax:0.4f}\n")
#     f.write(f"\n*** T = {t_semax}; K = {n_neighbors}; SeMax = {semax:0.4f}\n")
#     f.write(f"\n===================================================================================================================\n")
#     return weight


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

# def train(data_name, test_percent):
def train():
    # test_percent = int(test_percent)
    # data_path = results_path + "/" + data_name
    # print(data_path)
    # data = pd.read_csv(data_path)
    # data = data.drop_duplicates(subset=['CommonNeighbor', 'AdamicAdar', 'JaccardCoefficient', 'PreferentialAttachment', 'ResourceAllocation', 'ShortestPath' ,'CommonCountry', 'Label'])
    
    # X = data.drop(columns=['id_author_1', 'id_author_2', 'Label'])
    # y = data['Label']
    # print(X.shape)
    # print("Tỉ lệ nhãn -1-1")
    # print(np.sum(y==-1), end='--')
    # print(np.sum(y==1))

    # scaler = StandardScaler().fit(X)
    # X = scaler.transform(X)
    # Note: this is only for the purpose of demo. 
    # To be precise, train set and test set should be created by splitting the whole papers into two consecutive sets
    # For example: train set consists of papers from 2000-2008
    # test set consists of papers from 2009-2017
    # Then code should be like this
    data_train = pd.read_csv(results_path + "/" + "Data_22_2000_2010_unweighted_2009-12-31_static.csv")
    data_test = pd.read_csv(results_path + "/" + "Data_22_2010_2017_unweighted_2016-12-31_static.csv")

    data_train = data_train.drop_duplicates(subset=['CommonNeighbor', 'AdamicAdar', 'JaccardCoefficient', 'PreferentialAttachment', 'ResourceAllocation', 'ShortestPath' ,'CommonCountry', 'Label'])
    data_test = data_test.drop_duplicates(subset=['CommonNeighbor', 'AdamicAdar', 'JaccardCoefficient', 'PreferentialAttachment', 'ResourceAllocation', 'ShortestPath' ,'CommonCountry', 'Label'])

    X_train = data_train.drop(columns=['id_author_1', 'id_author_2', 'Label', 'CommonNeighbor', 'AdamicAdar', 'JaccardCoefficient', 'PreferentialAttachment', 'ResourceAllocation', 'ShortestPath'])
    y_train = data_train['Label']

    X_test = data_test.drop(columns=['id_author_1', 'id_author_2', 'Label', 'CommonNeighbor', 'AdamicAdar', 'JaccardCoefficient', 'PreferentialAttachment', 'ResourceAllocation', 'ShortestPath'])
    y_test = data_test['Label'] 

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_percent/100), random_state=608, shuffle=True, stratify=y)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True, stratify=y)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    # print(X_train[:10])
    C = 100

    thamso1 = 1
    thamso2 = 1
    T = 1
    N = 1
    n_neighbor = 5
    test_size = [0.2,0.3,0.4]
    testsize_val = 0.2
    K_big = 5
    K_small = 5
    # data = [Co_Author, Abanole, Ecoli, Ecloli1, Ecoli3, Glass1, Glass4, Haberman, Waveform, New_thyroid2, Page_blocks,
    #             Pima_Indians_Diabetes, Satimage, Transfusion, Yeast]

    #Haberman dataset
    # dataset = Haberman_KFold
    # beta_center, beta_estimate, beta_actual = 1, 1, 0.6 # !!!!!!! Beta with Dataset, change Data please change Beta !!!!!!!!

    # Ecoli dataset
    # dataset = Co_Author
    beta_center, beta_estimate, beta_actual = 0.3, 0.6, 0.7

    #Satimage dataset
    # dataset = Satimage_KFold
    # beta_center, beta_estimate, beta_actual = 0.9, 0.8, 0.2

    # Yeast dataset
    # dataset = Yeast_KFold
    # beta_center, beta_estimate, beta_actual = 1, 1, 0.4

    name_method =["own_class_center","estimated_hyper_lin","actual_hyper_lin","distance_center_own_opposite_tam"]
    #name_method =["own_class_center_divided"]
    name_function = ["lin_center_own","exp","func_own_opp_new"]

    # time = datetime.now().strftime("%d%m%Y_%H%M%S")
    # filepath = "D:/MULTIMEDIA/MACHINE_LEARNING_THAY_QUANG/DOCUMENTS/NC1/Compression/Dongtacgia/Project3/Project3_New/BE/text_script"
    # filename = (str(dataset).split("\\")[-1]).split(".")[0]
    # f = open(f"{filepath}/Data_{filename}_{time}_Detail.txt", "w")
    # f2 = open(f"{filepath}/Data_{filename}_{time}_Main.txt", "w")
    # for n in range(0,N):
    #     print("Lan boc: ",n+1)
    #     # for dataset in data:
    #     filename = (str(dataset).split("\\")[-1]).split(".")[0]
    #     f = open(f"{filepath}/Data_{filename}_{time}_Detail.txt", "w")
    #     f2 = open(f"{filepath}/Data_{filename}_{time}_Main.txt", "w")
        

        #print(f"\n\tUSING DATASET : {filename}\n")
        # for testsize in test_size:
        #     X_train, y_train, X_test, y_test = dataset.load_data(test_size=testsize)
        #X, y = dataset.load_data()

        # kfold_validation = StratifiedKFold(n_splits=5, shuffle=True)
        # header = ['Times','Fold','Name Method', 'Name Function', 'SP', 'SE', 'Gmean', 'F1 Score','Accuracy','AUC','Ma tran nham lan']
        # data = []
        # with open(f'D:/MULTIMEDIA/MACHINE_LEARNING_THAY_QUANG/DOCUMENTS/NC1/Compression/Dongtacgia/Project3/Project3_New/BE/Experiment/Data_{filename}_{time}_Main.csv', 'a', encoding='UTF8', newline='') as f3:
        #     writer = csv.writer(f3)
        #     writer.writerow(header)
        #     fold = 1
        #     for train_index, test_index in kfold_validation.split(X,y):
                
        #         X_train, y_train = X[train_index], y.iloc[train_index]
        #         X_test, y_test = X[test_index], y.iloc[test_index]
        #         #Scalling Data
        #         sc_X = StandardScaler()
        #         X_train = sc_X.fit_transform(X_train)
        #         X_test = sc_X.transform(X_test)
        #         y_train = np.array(y_train)

        #         #FuzyyWsvm
        #         for namemethod in name_method:
        #             for namefunction in name_function:
        #                 if namemethod =="distance_center_own_opposite_tam" and namefunction =="lin_center_own":
        #                     continue
        #                 elif namemethod =="distance_center_own_opposite_tam" and namefunction =="exp":
        #                     continue
        #                 elif namemethod == "own_class_center" and namefunction == "func_own_opp_new":
        #                     continue
        #                 elif namemethod == "estimated_hyper_lin" and namefunction == "func_own_opp_new":
        #                     continue
        #                 elif namemethod == "actual_hyper_lin" and namefunction == "func_own_opp_new":
        #                     continue
        #                 # elif namemethod == "distance_center_own_opposite_tam" and namefunction == "lin":
        #                 #     continue
        #                 # elif namemethod == "own_class_center" and namefunction == "lin":
        #                 #     continue
        #                 # elif namemethod == "estimated_hyper_lin" and namefunction == "lin":
        #                 #     continue
        #                 # elif namemethod == "actual_hyper_lin" and namefunction == "lin_center_own":
        #                 #     continue
        #                 else:
        #                     f.write(f"\n\n\tFuzzy W.SVM name_method = '{namemethod}',name_function = '{namefunction}' starting...\n")
        #                     f2.write(f"\n\n\tFuzzy W.SVM name_method = '{namemethod}',name_function = '{namefunction}' starting...\n")
        #                     print(f"Fuzzy W.SVM name_method = '{namemethod}',name_function = '{namefunction}' starting...\n")
        #                     distribution_weight = fuzzy_weight(f,beta_center, beta_estimate, beta_actual,X_train, y_train,namemethod,namefunction)
        #                     fold_a = f"{fold}_a"
        #                     for i in range(0,T):
        #                         new_W = lfb(f,C,distribution_weight,namemethod,namefunction,T,X_test,y_test,X_train,y_train,n_neighbor,thamso1,thamso2)
        #                         test_pred = wsvm(C,models_path,X_train, y_train, X_test,time,namefunction,namemethod, new_W)
        #                         sp,se,gmean = Gmean(y_test,test_pred)
        #                         fold_as=f"{fold_a}_{i+1}"
        #                         data.append([n,fold_as,namemethod,namefunction,sp,se,gmean,f1_score(y_test, test_pred),accuracy_score(y_test,test_pred),roc_auc_score(y_test, test_pred),str(confusion_matrix(y_test, test_pred))])
        #         fold = fold + 1
        #     writer.writerows(data)

    # for namemethod in name_method:
    #     for namefunction in name_function:
    #         if namemethod =="distance_center_own_opposite_tam" and namefunction =="lin_center_own":
    #             continue
    #         elif namemethod =="distance_center_own_opposite_tam" and namefunction =="exp":
    #             continue
    #         elif namemethod == "own_class_center" and namefunction == "func_own_opp_new":
    #             continue
    #         elif namemethod == "estimated_hyper_lin" and namefunction == "func_own_opp_new":
    #             continue
    #         elif namemethod == "actual_hyper_lin" and namefunction == "func_own_opp_new":
    #             continue
    #         # elif namemethod == "distance_center_own_opposite_tam" and namefunction == "lin":
    #         #     continue
    #         # elif namemethod == "own_class_center" and namefunction == "lin":
    #         #     continue
    #         # elif namemethod == "estimated_hyper_lin" and namefunction == "lin":
    #         #     continue
    #         # elif namemethod == "actual_hyper_lin" and namefunction == "lin_center_own":
    #         #    continue
    #         # elif namemethod == "own_class_center" and namefunction == "lin_center_own":
    #         #     continue 
    #         # elif namemethod == "own_class_center" and namefunction == "exp":
    #         #     continue 
    #         # elif namemethod == "estimated_hyper_lin" and namefunction == "lin_center_own":
    #         #     continue 
    #         # elif namemethod == "estimated_hyper_lin" and namefunction == "exp":
    #         #     continue 
    #         # elif namemethod == "actual_hyper_lin" and namefunction == "lin_center_own":
    #         #     continue
    #         # elif namemethod == "actual_hyper_lin" and namefunction == "exp":
    #         #     continue
    #         else:
    #             f.write(f"\n\n\tFuzzy W.SVM name_method = '{namemethod}',name_function = '{namefunction}' starting...\n")
    #             f2.write(f"\n\n\tFuzzy W.SVM name_method = '{namemethod}',name_function = '{namefunction}' starting...\n")
    #             print(f"Fuzzy W.SVM name_method = '{namemethod}',name_function = '{namefunction}' starting...\n")
    #             distribution_weight = fuzzy_weight(f,beta_center, beta_estimate, beta_actual,X_train, y_train,namemethod,namefunction)
    #             test_pred = wsvm(C,models_path,X_train, y_train, X_test,time,namefunction,namemethod,distribution_weight)
    #             sp,se,gmean = Gmean(y_test,test_pred)
    #             metr_text(f,X_train,y_test,test_pred,sp,se,gmean)
    # #             metr_text(f2,X_train,y_test,test_pred,sp,se,gmean)

    # model2 = SVC(kernel='rbf', max_iter=5000)
    model2=SVC(probability=True, kernel='linear')
    model2.fit(X_train, y_train)

    y_pred = model2.predict(X_test)

    print("Precision:",metrics.precision_score(y_test, y_pred))
    print("Recall:",metrics.recall_score(y_test, y_pred))
    print("F1 score:",metrics.f1_score(y_test, y_pred))
    print("Roc_auc_score score:",metrics.roc_auc_score(y_test, y_pred))
    
    result = {}
    result['Precision'] = metrics.precision_score(y_test, y_pred)
    result['Recall'] = metrics.recall_score(y_test, y_pred)
    result['f1_score'] = metrics.f1_score(y_test, y_pred)
    result['Roc_auc'] = metrics.roc_auc_score(y_test, y_pred)

    tmp = data_train.split('_')
    
    tmp[0] = "Model"
    model_name = "_".join(tmp)[:-4] + ".pkl"
    with open(models_path + '/' + model_name, 'wb') as file:
        pickle.dump(model2, file)

    # tmp[0] = "Scaler"
    # scaler_name = "_".join(tmp)[:-4] + ".pkl"
    # with open(models_path + '/' + scaler_name, 'wb') as file:
    #     pickle.dump(scaler, file)

    return json.dumps({"results": result, "model_name": model_name})


train()