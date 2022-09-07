'''
This script contains functions for to conduct experiments
Author : Sai Venkata Krishnaveni,Devarakonda
Date : 05/06/2022
'''

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from utilities import Load_image,Load_lxyr,knn_classifier,accuracy,data_split,PCA,crop_images,class_wise_accuracy, KNN_train,k_fold_split,k_fold_split_equal
import numpy as np

#cropping
ls_train_x,ls_train_y = crop_images()

'''
#knn
for k in [5]:
    preds_train = []
    for i in range(len(arr_train_pc_x)):
        n,t = knn_classifier(arr_train_pc_x,arr_train_pc_y,arr_train_pc_x[i],k)
        elt = max(set(t), key=t.count)
        preds_train.append(elt)
    unq_train_preds, cnt_train_preds = np.unique(preds_train, return_counts=True)
    train_acc = accuracy(arr_train_pc_y,preds_train)

    preds_test = []
    for i in range(len(arr_test_pc_x)):
        n,t = knn_classifier(arr_train_pc_x,arr_train_pc_y,arr_test_pc_x[i],k)
        elt = max(set(t), key=t.count)
        preds_test.append(elt)
    unq_test_preds, cnt_test_preds = np.unique(preds_test, return_counts=True)
    test_acc = accuracy(arr_test_pc_y,preds_test)

    print('Train Accuracy  with k=  ' + str(k) + ' is ' + str(train_acc))
    print('Test Accuracy with k=  ' + str(k) + ' is ' + str(test_acc))
'''
'''
preds_test = []
for i in range(len(arr_test_pc_x)):
    n,t = knn_classifier(arr_train_pc_x,arr_train_pc_y,arr_test_pc_x[i],k)
    elt = max(set(t), key=t.count)
    preds_test.append(elt)
unq_test_preds, cnt_test_preds = np.unique(preds_test, return_counts=True)
test_acc = accuracy(arr_test_pc_y,preds_test)

print('Train Accuracy  with k=  ' + str(k) + ' is ' + str(train_acc))
print('Test Accuracy with k=  ' + str(k) + ' is ' + str(test_acc))
'''


######################################################################################### Experiments #########################################################################################################

#Number of principle components  = 200, image cropping size = (25,25), with one train and test split (75%,25%)
pc,arr_y,var_perc = PCA(ls_train_x,ls_train_y,200)

arr_train_pc_x,arr_train_pc_y,arr_test_pc_x,arr_test_pc_y= data_split(pc,arr_y,75,200)
unp_train, cnt_train = np.unique(arr_train_pc_y, return_counts=True)
unp_test, cnt_test = np.unique(arr_test_pc_y, return_counts=True)
preds_train = []
for k in [1,2,3,4,5,6,7]:

    acc1,acc2,acc3,acc4 = KNN_train(k,arr_train_pc_x,arr_train_pc_y,arr_train_pc_x,arr_test_pc_y,Train=True)
    print('Train Accuracy for label 1 with k=  ' + str(k) + ' is ' + str(acc1))
    print('Train Accuracy for label 2 with k=  ' + str(k) + ' is ' + str(acc2))
    print('Train Accuracy for label 3 with k=  ' + str(k) + ' is ' + str(acc3))
    print('Train Accuracy for label 4 with k=  ' + str(k) + ' is ' + str(acc4))


#Number of principle components  = 200, image cropping size = (25,25), with 5-fold(imbalance classes) cross validation
#code for PCA
pc,arr_y,var_perc = PCA(ls_train_x,ls_train_y,200)
for k in [1,2,3,4,5,6,7]:
    sum_acc1 = 0
    sum_acc2 = 0
    sum_acc3 = 0
    sum_acc4 = 0
    for n in [1,2,3,4,5]:
        train_data,test_data = k_fold_split(pc,arr_y,n)
        #print(np.unique(train_data[:,-1],return_counts= True)[1])
        #print(np.unique(test_data[:, -1], return_counts=True)[1])
        arr_train_x = train_data[:,:-1]
        arr_train_y = train_data[:,-1]
        arr_test_x = test_data[:,:-1]
        arr_test_y = test_data[:,-1]

        preds_train = []

        acc1,acc2,acc3,acc4 = KNN_train(k,arr_train_x,arr_train_y,arr_test_x,arr_test_y,Train=False)
        #print('Train Accuracy for label 1 with k=  ' + str(k) + ' is ' + str(acc1))
        #print('Train Accuracy for label 2 with k=  ' + str(k) + ' is ' + str(acc2))
        #print('Train Accuracy for label 3 with k=  ' + str(k) + ' is ' + str(acc3))
        #print('Train Accuracy for label 4 with k=  ' + str(k) + ' is ' + str(acc4))
        #print()
        sum_acc1 = sum_acc1+acc1
        sum_acc2 = sum_acc2 + acc2
        sum_acc3 = sum_acc3 + acc3
        sum_acc4 = sum_acc4 + acc4

    avg_acc1 = sum_acc1/5
    avg_acc2 = sum_acc2/5
    avg_acc3 = sum_acc3/5
    avg_acc4 = sum_acc4/5

    print('average Accuracy for label 1 with k=  ' + str(k) + ' is ' + str(avg_acc1))
    print('average Accuracy for label 2 with k=  ' + str(k) + ' is ' + str(avg_acc2))
    print('average Accuracy for label 3 with k=  ' + str(k) + ' is ' + str(avg_acc3))
    print('average Accuracy for label 4 with k=  ' + str(k) + ' is ' + str(avg_acc4))
    print()

#Number of principle components  = 200, image cropping size = (25,25), with 5-fold(equal sizes) cross validation
#code for PCA
pc,arr_y,var_perc = PCA(ls_train_x,ls_train_y,200)
for k in [1,2,3,4,5,6,7]:
    sum_acc1 = 0
    sum_acc2 = 0
    sum_acc3 = 0
    sum_acc4 = 0
    for n in [1,2,3,4,5]:
        train_data,test_data = k_fold_split_equal(pc,arr_y,n)
        #print(np.unique(train_data[:,-1],return_counts= True)[1])
        #print(np.unique(test_data[:, -1], return_counts=True)[1])
        arr_train_x = train_data[:,:-1]
        arr_train_y = train_data[:,-1]
        arr_test_x = test_data[:,:-1]
        arr_test_y = test_data[:,-1]

        preds_train = []

        acc1,acc2,acc3,acc4 = KNN_train(k,arr_train_x,arr_train_y,arr_test_x,arr_test_y,Train=False)
        #print('Train Accuracy for label 1 with k=  ' + str(k) + ' is ' + str(acc1))
        #print('Train Accuracy for label 2 with k=  ' + str(k) + ' is ' + str(acc2))
        #print('Train Accuracy for label 3 with k=  ' + str(k) + ' is ' + str(acc3))
        #print('Train Accuracy for label 4 with k=  ' + str(k) + ' is ' + str(acc4))
        #print()
        sum_acc1 = sum_acc1+acc1
        sum_acc2 = sum_acc2 + acc2
        sum_acc3 = sum_acc3 + acc3
        sum_acc4 = sum_acc4 + acc4

    avg_acc1 = sum_acc1/5
    avg_acc2 = sum_acc2/5
    avg_acc3 = sum_acc3/5
    avg_acc4 = sum_acc4/5

    print('average Accuracy for label 1 with k=  ' + str(k) + ' is ' + str(avg_acc1))
    print('average Accuracy for label 2 with k=  ' + str(k) + ' is ' + str(avg_acc2))
    print('average Accuracy for label 3 with k=  ' + str(k) + ' is ' + str(avg_acc3))
    print('average Accuracy for label 4 with k=  ' + str(k) + ' is ' + str(avg_acc4))
    print()


#K value in KNN classifier = 2, image cropping size = (25,25), with 5-fold(equal sizes) cross validation
for p in [2,5,15,25,35,45,70,200]:
    k=2
    # code for PCA
    pc, arr_y, var_perc = PCA(ls_train_x, ls_train_y, p)
    sum_acc1 = 0
    sum_acc2 = 0
    sum_acc3 = 0
    sum_acc4 = 0
    for n in [1,2,3,4,5]:
        train_data,test_data = k_fold_split_equal(pc,arr_y,n)
        #print(np.unique(train_data[:,-1],return_counts= True)[1])
        #print(np.unique(test_data[:, -1], return_counts=True)[1])
        arr_train_x = train_data[:,:-1]
        arr_train_y = train_data[:,-1]
        arr_test_x = test_data[:,:-1]
        arr_test_y = test_data[:,-1]

        preds_train = []

        acc1,acc2,acc3,acc4 = KNN_train(k,arr_train_x,arr_train_y,arr_test_x,arr_test_y,Train=False)
        #print('Train Accuracy for label 1 with k=  ' + str(k) + ' is ' + str(acc1))
        #print('Train Accuracy for label 2 with k=  ' + str(k) + ' is ' + str(acc2))
        #print('Train Accuracy for label 3 with k=  ' + str(k) + ' is ' + str(acc3))
        #print('Train Accuracy for label 4 with k=  ' + str(k) + ' is ' + str(acc4))
        #print()
        sum_acc1 = sum_acc1+acc1
        sum_acc2 = sum_acc2 + acc2
        sum_acc3 = sum_acc3 + acc3
        sum_acc4 = sum_acc4 + acc4

    avg_acc1 = sum_acc1/5
    avg_acc2 = sum_acc2/5
    avg_acc3 = sum_acc3/5
    avg_acc4 = sum_acc4/5

    print('average Accuracy for label 1 with k=  ' + str(k) + ' is ' + str(avg_acc1))
    print('average Accuracy for label 2 with k=  ' + str(k) + ' is ' + str(avg_acc2))
    print('average Accuracy for label 3 with k=  ' + str(k) + ' is ' + str(avg_acc3))
    print('average Accuracy for label 4 with k=  ' + str(k) + ' is ' + str(avg_acc4))
    print()
