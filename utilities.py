'''
This script contains functions for loading images,cropping images,classifier and PCA
Author : Sai Venkata Krishnaveni,Devarakonda
Date : 05/06/2022
'''

from skimage.transform import resize
import numpy as np
import glob


############################################################ Load images ##########################################################
def Load_image(str_image_path):
    with open(str_image_path, mode='rb') as f:
        d = np.fromfile(f, dtype=np.uint8, count=1024 * 1024).reshape(1024, 1024)
    return d

############################################################# load labels ##########################################################
def Load_lxyr(str_lxyr_path):
    with open(str_lxyr_path) as f:
        d = np.fromfile(f,sep=' ')
    row = d.shape[0]/4
    d_new = d.reshape(int(row),4)
    return d_new

############################################################## crop images ##########################################################
def crop_images():
    img_path = '.\Images/*.sdt'
    ls_train_x = []
    ls_train_y = []
    radius_list = np.array([])
    labels_list = np.array([])
    for filename in glob.glob(img_path):
        arr_img = Load_image(filename)
        subj_name = filename.split("\\")[-1][:-4]
        lxyr_path = '.\GroundTruths\\' + subj_name + '.lxyr'
        d = Load_lxyr(lxyr_path)
        # check if volcanoes are present
        if np.any(d):
            label = d[:, 0]
            x = d[:, 1]
            x = x.astype(int)
            y = d[:, 2]
            y = y.astype(int)
            r = d[:, 3]
            r = r.astype(int)
            radius_list = np.append(radius_list, r)
            labels_list = np.append(labels_list, label)
            for ll, xx, yy, rr in zip(label, x, y, r):
                if ~((xx < rr) or (yy < rr)):
                    cropped_img = arr_img[yy - int(rr):yy + int(rr), xx - int(rr):xx + int(rr)]
                    # cropped_img_reshaped = np.resize(cropped_img,[25,25])
                    cropped_img_reshaped = resize(cropped_img, [25, 25])
                    ls_train_x.append(cropped_img_reshaped.ravel())
                    ls_train_y.append(ll)
    return ls_train_x,ls_train_y

############################################################## dimensionality reduction with PCA ##########################################################
def PCA(ls_train_x,ls_train_y,pca_elts):
    #features and corresponding labels
    arr_train_x= np.asarray(ls_train_x)
    arr_y= np.asarray(ls_train_y)

    #Normalize
    X = arr_train_x - arr_train_x.mean(axis=0)
    Z = X/arr_train_x.std(axis=0)
    #computing covariance matrix
    Z = np.dot(Z.T, Z)
    eigenvalues, eigenvectors = np.linalg.eig(Z)
    D = np.diag(eigenvalues)
    P = eigenvectors

    v_percent = [i/np.sum(eigenvalues) for i in eigenvalues]
    #choosing principle components based on highest eigen values
    pc = np.dot(arr_train_x,P[:,0:pca_elts])
    return pc,arr_y, v_percent*100


############################################################## KNN classifier ##########################################################
def cartesian_distance(a,b):
    '''
    This function calculates euclidean distance between 2 vectors
    inputs:Two vectors
    outputs:Euclidean Distance between given vectors
    '''
    distance=np.sqrt(np.sum(np.square(a-b)))
    return distance

#Function for KNN classifier which returns distance to k neighbors and their tagets for given test sample
def knn_classifier(train_data,train_labels,sample,k):
    '''
    This function is KNN classifier
    inputs:
        train_features = mxn array (m= #observations,n= #features)
        train_labels = nx1 array of targets
        sample = 1 row of test features
        k =  int (#nearest neighbors)
    outputs:
        Euclidean distance of k neighbors
        Targets of k nearest neighbors
    '''
    distance=[]
    neighbors=[]
    for i in range(len(train_data)):
        d=cartesian_distance(train_data[i],sample)
        distance.append(d)
    ind = np.argsort(distance)
    distance.sort()
    for i in range(k):
        neighbors.append(distance[i])
    targets = [train_labels[i] for i in ind[:k]]
    return neighbors,targets


############################################################## performance functions ##########################################################
def accuracy(y_true, y_pred):
    '''
    This function calculates accuracy
    Input : actual and expected labels
    Output : accuracy
    '''
    count = 0
    for i in range(len(y_true)):
        if (y_true[i] == y_pred[i]):
            count = count + 1
    accuracy = count / len(y_true) * 100
    return accuracy


#performance function(k-fold)
def class_wise_accuracy(y_true, y_pred):
    '''
    This function calculates accuracy
    Input : actual and expected labels
    Output : accuracy
    '''
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    unq_y_true, cnt_y_true = np.unique(y_true, return_counts=True)
    for i in range(len(y_true)):
        if (y_true[i] == y_pred[i]):
            if(y_pred[i]==1.):
                count1 = count1 + 1
            if (y_pred[i] == 2.):
                count2 = count2 + 1
            if (y_pred[i] == 3.):
                count3 = count3 + 1
            if (y_pred[i] == 4.):
                count4 = count4 + 1
    accuracy1 = count1 / cnt_y_true[0] * 100
    accuracy2 = count2 / cnt_y_true[1] * 100
    accuracy3 = count3 / cnt_y_true[2] * 100
    accuracy4 = count4 / cnt_y_true[3] * 100

    return accuracy1,accuracy2,accuracy3,accuracy4


############################################################## spliiting train data and test data ##########################################################

def data_split(pc,arr_y,split,pca_elts):
    arr_y_rshp = arr_y.reshape(len(arr_y), 1)
    pcdata_and_labels = np.append(pc, arr_y_rshp, axis=1)
    unq_full_data, count_full_data = np.unique(pcdata_and_labels[:, pca_elts], return_counts=True)
    arr_train_pc_x = []
    arr_train_pc_y = []
    arr_test_pc_x = []
    arr_test_pc_y = []
    for i in range(len(unq_full_data)):
        label_ind = np.argwhere(pcdata_and_labels == unq_full_data[i])
        arr_pc = pcdata_and_labels[label_ind[:, 0]]
        arr_train_x = arr_pc[0:int(len(arr_pc) * split / 100), :-1]
        arr_train_y = arr_pc[0:int(len(arr_pc) * split / 100), -1]

        arr_test_x = arr_pc[int(len(arr_pc) * split / 100):, :-1]
        arr_test_y = arr_pc[int(len(arr_pc) * split / 100):, -1]

        arr_train_pc_x.extend(arr_train_x)
        arr_train_pc_y.extend(arr_train_y)
        arr_test_pc_x.extend(arr_test_x)
        arr_test_pc_y.extend(arr_test_y)

    arr_train_pc_x = np.array(arr_train_pc_x)
    arr_train_pc_y = np.array(arr_train_pc_y)
    arr_test_pc_x = np.array(arr_test_pc_x)
    arr_test_pc_y = np.array(arr_test_pc_y)
    return arr_train_pc_x,arr_train_pc_y,arr_test_pc_x,arr_test_pc_y

def k_fold_split(pc,arr_y,n):
    #code for k-fold
    k = 5
    arr_y_rshp = arr_y.reshape(len(arr_y), 1)
    pcdata_and_labels = np.append(pc, arr_y_rshp, axis=1)
    uniq, cont = np.unique(pcdata_and_labels[:,-1], return_counts=True)
    arrx_fold_1 = []
    arrx_fold_2 = []
    arrx_fold_3 = []
    arrx_fold_4 = []
    arrx_fold_5 = []

    for i in range(len(uniq)):
        label_ind = np.argwhere(pcdata_and_labels == uniq[i])
        arr_all_data = pcdata_and_labels[label_ind[:, 0]]
        length= int(len(label_ind) / k)
        #print(length)
        for j in range(k):
            arrx = arr_all_data[j * length:(j + 1) * length]
            if(j==0):
                arrx_fold_1.extend(arrx)
            if(j==1):
                arrx_fold_2.extend(arrx)
            if(j==2):
                arrx_fold_3.extend(arrx)
            if(j==3):
                arrx_fold_4.extend(arrx)
            if(j ==4):
                arrx_fold_5.extend(arrx)

    for k in range(1,k+1):
        ls = [1,2,3,4,5]
        test_foldX = eval('arrx_fold_'+str(k))
        test_foldX = np.array(test_foldX)
        ls.remove(k)
        #print(ls)
        train_foldX = np.concatenate((eval('arrx_fold_'+str(ls[0])),eval('arrx_fold_'+str(ls[1])),eval('arrx_fold_'+str(ls[2])),eval('arrx_fold_'+str(ls[3]))))
        if(n==k):
            return train_foldX,test_foldX



def k_fold_split_equal(pc,arr_y,n):
    k = 5
    arr_y_rshp = arr_y.reshape(len(arr_y), 1)
    pcdata_and_labels = np.append(pc, arr_y_rshp, axis=1)
    uniq, cont = np.unique(pcdata_and_labels[:,-1], return_counts=True)
    arrx_fold_1 = []
    arrx_fold_2 = []
    arrx_fold_3 = []
    arrx_fold_4 = []
    arrx_fold_5 = []

    for i in range(len(uniq)):
        label_ind = np.argwhere(pcdata_and_labels == uniq[i])
        arr_all_data = pcdata_and_labels[label_ind[:, 0]]
        arr_all_data = arr_all_data[-140:]
        length= int(len(arr_all_data) / k)
        for j in range(k):
            arrx = arr_all_data[j * length:(j + 1) * length]
            if(j==0):
                arrx_fold_1.extend(arrx)
            if(j==1):
                arrx_fold_2.extend(arrx)
            if(j==2):
                arrx_fold_3.extend(arrx)
            if(j==3):
                arrx_fold_4.extend(arrx)
            if(j ==4):
                arrx_fold_5.extend(arrx)

    for x in range(1,k+1):
        ls = [1,2,3,4,5]
        test_foldX = eval('arrx_fold_'+str(x))
        test_foldX = np.array(test_foldX)
        ls.remove(x)
        #print(ls)
        train_foldX = np.concatenate((eval('arrx_fold_'+str(ls[0])),eval('arrx_fold_'+str(ls[1])),eval('arrx_fold_'+str(ls[2])),eval('arrx_fold_'+str(ls[3]))))
        if(n==x):
            return train_foldX,test_foldX

############################################################## functions for experiments ##########################################################
def KNN_train(k, arr_train_pc_x,arr_train_pc_y,test_data,arr_test_pc_y, Train=False):
    preds = []
    for i in range(len(test_data)):
        n, t = knn_classifier(arr_train_pc_x, arr_train_pc_y, test_data[i], k)
        unp_t, cnt_t = np.unique(t, return_counts=True)
        unp_train, cnt_train = np.unique(arr_train_pc_x, return_counts=True)
        ind1 = unp_t - 1
        ind1 = ind1.astype(int)
        e = cnt_t / cnt_train[ind1]
        #e = cnt_t
        max_ind = np.argmax(e)
        # print(unp_t[max_ind])
        elt = unp_t[max_ind]
        preds.append(elt)
    if Train:
        acc1, acc2, acc3, acc4 = class_wise_accuracy(arr_train_pc_y, preds)
    else:
        acc1, acc2, acc3, acc4 = class_wise_accuracy(arr_test_pc_y, preds)

    return acc1,acc2,acc3,acc4


