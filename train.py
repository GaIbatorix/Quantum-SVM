 # Created by Gabriele Cavallaro (g.cavallaro@fz-juelich.de) 

import sys
import os

import numpy as np
from sklearn.model_selection import KFold
from sklearn import preprocessing

from utils import *
from quantum_SVM import *


def main():
    
    path_dataset=sys.argv[1] # Path+data (file.txt)
    id_dataset=sys.argv[1][len(sys.argv[1])-8:-4] 
    experiments=int(sys.argv[2])
    
    # These paths should be created in advance 
    train_path='input_datasets/train/'+id_dataset+'/'  # path_data_key
    train_key = id_dataset+'calibtrain'
    train_out='outputs/train/'+id_dataset+'/'
    calibration_out='outputs/train/'+id_dataset+'/'   
    
    # Load the data
    [X_train, Y_train, X_test]=dataread(path_dataset)
   
    slice=40 # Number of samples to use for the training
    fold=int(len(X_train)/slice)

    for i in range(0,experiments):    
        cv = KFold(n_splits=fold, random_state=i, shuffle=True)
        count=0
        for test_index, train_index in cv.split(X_train):
            #print("Train Index: ", len(train_index), "\n")
            X_train_slice, y_train_slice = X_train[train_index], Y_train[train_index]
            X_train_slice = preprocessing.scale(X_train_slice)
            write_samples(X_train_slice, y_train_slice,f''+train_path+'/'+train_key+str(i)+'_'+str(count))
            count=count+1

    print("Each training set has", len(train_index), "samples\n")
    
    
    # Get the calibration results
    hyperparameters=np.load(calibration_out+'hyperparameters.npy')
    testauprc_all=np.load(calibration_out+'testauprc_all.npy')

    # Select the best hyperparameter set for the max value of testauprc
    idx_max = np.where(testauprc_all == np.amax(testauprc_all))
    B=int(hyperparameters[int(idx_max[0]),0])
    K=int(hyperparameters[int(idx_max[0]),1])
    xi=int(hyperparameters[int(idx_max[0]),2])
    gamma=hyperparameters[int(idx_max[0]),3]
    print('The best hyperparameters are:\n'+'B = '+str(B)+' K = '+str(K)+' xi = '+str(xi)+' gamma = '+str(gamma))


    trained_SVMs=[]

    for j in range(0,experiments):
        for i in range(0,fold):
            path=gen_svm_qubos(B,K,xi,gamma,train_path,train_key+str(j)+'_'+str(i),train_out)
            trained_SVMs.append(dwave_run(train_path,path))
        
    np.save(train_out+'trained_SVMs',trained_SVMs)   
    
    

    

# In the path where the code is placed, create the following folders
#  -> input_datasets/train/id_dataset (e.g., id_dataset: Im40 )
#  -> outputs/train/id_dataset

# How to run the train
# -> python train.py dataset.txt number_of_experiments (e.g., python train.py Im40.txt 1)
# Where the input parameters are
# -> dataset.txt: the name of the file (e.g., Im40.txt, avaialble at http://hyperlabelme.uv.es/)
# -> number_of_experiments: the number of runs of each training experiment  


if __name__ == "__main__":
    
    if len(sys.argv) < 3:
         print('Usage: '+sys.argv[0]+' dataset.txt number_of_experiments')

    main()
    exit(-1)
