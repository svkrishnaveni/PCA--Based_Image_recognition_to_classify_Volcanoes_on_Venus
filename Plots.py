'''
This script contains plots which are helpful for data exploration of volcanic image pathces
Author : Sai Venkata Krishnaveni,Devarakonda
Date : 05/08/2022
'''

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from utilities import Load_image,Load_lxyr,knn_classifier,accuracy, PCA,crop_images
import glob
import numpy as np
from skimage.transform import resize
plt.close('all')

#Overlay circles on the grayscale image showing volcanoes spatial presence on two example images.
def circle_on_one_volcano():
    filename ='.\img1.sdt'
    arr_img = Load_image(filename)
    lxyr_path = './img1.lxyr'
    img = Load_image(filename)
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    subj_name = filename.split("\\")[-1][:-4]
    d = Load_lxyr(lxyr_path)
    x = d[:,1]
    y = d[:,2]
    r = d[:,3]
    for xx,yy,rr in zip(x,y,r):
        circ = Circle((xx,yy),rr)
        ax.add_patch(circ)
    # Show the image
    plt.imshow(img,cmap = 'gray')
    ax.invert_yaxis()
    plt.show()

#Overlay circles on the grayscale image showing volcanoes spatial presence for all images
def circle_on_all_volcano():
    for filename in glob.glob('.\Images/*.sdt'):
        img = Load_image(filename)
        fig,ax = plt.subplots(1)
        ax.set_aspect('equal')
        subj_name = filename.split("\\")[-1][:-4]
        lxyr_path = '.\GroundTruths\\' + subj_name + '.lxyr'

        d = Load_lxyr(lxyr_path)
        x = d[:,1]
        y = d[:,2]
        r = d[:,3]
        for xx,yy,rr in zip(x,y,r):
            circ = Circle((xx,yy),rr)
            ax.add_patch(circ)
        # Show the image
        plt.imshow(img,cmap = 'gray')
        ax.invert_yaxis()
        plt.show()


#Plot showing distribution of radius of the volcanoes
def radius_dist():
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
            labels_list = np.append(labels_list, label)
            for ll, xx, yy, rr in zip(label, x, y, r):
                if ~((xx < rr) or (yy < rr)):
                    cropped_img = arr_img[yy - int(rr):yy + int(rr), xx - int(rr):xx + int(rr)]
                    # cropped_img_reshaped = np.resize(cropped_img,[25,25])
                    cropped_img_reshaped = resize(cropped_img, [35, 35])
                    ls_train_x.append(cropped_img_reshaped.ravel())
                    ls_train_y.append(ll)
                    radius_list = np.append(radius_list, rr)
    plt.hist(radius_list,bins=300)
    plt.xlabel('Radius of volcano')
    plt.ylabel('Num of volcano image patches')
    plt.show()


#Histogram showing distribution of labels of volcanic image patches
def labels_dist():
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
            labels_list = np.append(labels_list, label)
            for ll, xx, yy, rr in zip(label, x, y, r):
                if ~((xx < rr) or (yy < rr)):
                    cropped_img = arr_img[yy - int(rr):yy + int(rr), xx - int(rr):xx + int(rr)]
                    # cropped_img_reshaped = np.resize(cropped_img,[25,25])
                    cropped_img_reshaped = resize(cropped_img, [35, 35])
                    ls_train_x.append(cropped_img_reshaped.ravel())
                    ls_train_y.append(ll)
                    radius_list = np.append(radius_list, rr)
    plt.hist(ls_train_y)
    plt.rcParams.update({'font.size': 32})
    plt.xlabel('labels -- certainty of volcanoes')
    plt.ylabel('Num of volcano image patches')
    plt.show()


#Plot showing percent variance explained versus number of principal components.
def plot_pca():
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
            labels_list = np.append(labels_list, label)
            for ll, xx, yy, rr in zip(label, x, y, r):
                if ~((xx < rr) or (yy < rr)):
                    cropped_img = arr_img[yy - int(rr):yy + int(rr), xx - int(rr):xx + int(rr)]
                    cropped_img_reshaped = resize(cropped_img, [25, 25])
                    ls_train_x.append(cropped_img_reshaped.ravel())
                    ls_train_y.append(ll)
                    radius_list = np.append(radius_list, rr)
    ls_var = np.array([])
    X1 = np.array([])
    PC, y, percent_var = PCA(ls_train_x, ls_train_y, 56)
    #for x in [2,5, 10,20,30,40,50,60,70,80,90,100,120,160]:
    for x in [2,5, 15,25,35,45,70,200]:
        var = np.sum(percent_var[:x])
        ls_var =np.append(ls_var,var)
        X1 =np.append(X1,x)
        #print(ls_var*100)
    plt.plot(X1, ls_var*100, 'go--', linewidth = 2, markersize=12)
    plt.xlabel('Number of principal components')
    plt.ylabel('Percent variance explained')
    plt.show()



circle_on_one_volcano()
radius_dist()
labels_dist()
plot_pca()


#Plotting first two principal components (scatterplot)
ls_train_x,ls_train_y = crop_images()
pc,arr_y,var_perc = PCA(ls_train_x,ls_train_y,2)
plt.scatter(pc[:,0], pc[:,1], c=arr_y, alpha=1.0,cmap='gist_rainbow')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()