# Created by Gabriele Cavallaro (g.cavallaro@fz-juelich.de) 

import sys
import os

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from utils import *
from quantum_SVM import *


def main():
        
    path_dataset=sys.argv[1] # Path+data (file.txt)
    id_dataset=sys.argv[1][len(sys.argv[1])-8:-4] 
    fold=int(sys.argv[2])
    
    # These paths should be created in advance 
    calibration_path='input_datasets/calibration/'+id_dataset+'/'  # path_data_key
    calibration_key = id_dataset+'calibtrain'
    calibration_out='outputs/calibration/'+id_dataset+'/'
   
    # Load the data
    [X_train, Y_train, X_test]=dataread(path_dataset)
        
    
    for i in range(0,fold):
        X_train_cal, X_val_cal, Y_train_cal, Y_val_cal = train_test_split(X_train,Y_train, test_size=0.94, random_state=i)
    
        # Pre-processing 
        X_train_cal = preprocessing.scale(X_train_cal)
        X_val_cal = preprocessing.scale(X_val_cal)
        
        # Write the data
        write_samples(X_train_cal, Y_train_cal,calibration_path+id_dataset+'calibtrain'+str(i))
        write_samples(X_val_cal, Y_val_cal,calibration_path+id_dataset+'calibval'+str(i))
    
    print('Each training set includes '+str(X_train_cal.shape[0])+ ' samples')
    print('Each validation set includes '+str(X_val_cal.shape[0])+ ' samples')
    

    # Hyperparameters 
    B=[2,3,5,10]
    K=[2,3]
    xi=[0,1,5]
    gamma=[-1,0.125,0.25,0.5,1,2,4,8]

    n_experiments=len(B)*len(K)*len(xi)*len(gamma)

    hyperparameters=np.zeros([n_experiments,4], dtype=float)

    trainacc=np.zeros([fold], dtype=float)
    trainauroc=np.zeros([fold], dtype=float)
    trainauprc=np.zeros([fold], dtype=float)
    
    testacc=np.zeros([fold], dtype=float)
    testauroc=np.zeros([fold], dtype=float)
    testauprc=np.zeros([fold], dtype=float)

    trainacc_all=np.zeros([n_experiments], dtype=float)
    trainauroc_all=np.zeros([n_experiments], dtype=float)
    trainauprc_all=np.zeros([n_experiments], dtype=float)
    
    testacc_all=np.zeros([n_experiments], dtype=float)
    testauroc_all=np.zeros([n_experiments], dtype=float)
    testauprc_all=np.zeros([n_experiments], dtype=float)


    f = open(calibration_out+'calibration_results.txt',"w") 
    f.write("B\t K\t xi\t   gamma\t trainacc\t trainauroc\t trainauprc\t testacc\t testauroc\t testauprc\n") 
  
    count=0 
    for x in range(0,len(B)):
        for y in range(0,len(K)):
            for z in range(0,len(xi)):
                for i in range(0,len(gamma)):
                    for j in range(0,fold):
                        path=gen_svm_qubos(B[x],K[y],xi[z],gamma[i],calibration_path,calibration_key+str(j),calibration_out)
                        pathsub=dwave_run(calibration_path,path)
                        [trainacc[j],trainauroc[j],trainauprc[j],testacc[j],testauroc[j],testauprc[j]]=eval_run_rocpr_curves(calibration_path,pathsub,'noplotsave')
                    
                    hyperparameters[count,0]=B[x]
                    hyperparameters[count,1]=K[y]
                    hyperparameters[count,2]=xi[z]
                    hyperparameters[count,3]=gamma[i]
            
                    trainacc_all[count]=np.average(trainacc)
                    trainauroc_all[count]=np.average(trainauroc)
                    trainauprc_all[count]=np.average(trainauprc)
    
                    testacc_all[count]=np.average(testacc)
                    testauroc_all[count]=np.average(testauroc)
                    testauprc_all[count]=np.average(testauprc)
                
                    np.save(calibration_out+'hyperparameters', hyperparameters)
                    np.save(calibration_out+'trainacc_all', trainacc_all)
                    np.save(calibration_out+'trainauroc_all', trainauroc_all)
                    np.save(calibration_out+'trainauprc_all', trainauprc_all)
                    np.save(calibration_out+'testacc_all', testacc_all)
                    np.save(calibration_out+'testauroc_all', testauroc_all)
                    np.save(calibration_out+'testauprc_all', testauprc_all)
                
                    f.write(f'{B[x]}\t {K[y]}\t {xi[z]}\t {gamma[i]:8.3f}\t {np.average(trainacc):8.4f}\t {np.average(trainauroc):8.4f}\t {np.average(trainauprc):8.4f}\t {np.average(testacc):8.4f}\t {np.average(testauroc):8.4f}\t {np.average(testauprc):8.4f}')
                    f.write("\n") 
                    count=count+1
                
    f.close() 
    
    

# In the path where the code is placed, create the following folders
#  -> input_datasets/calibration/id_dataset (e.g., id_dataset: Im40 )
#  -> outputs/calibration/id_dataset

# How to run the calibration
# -> python calibration.py dataset.txt n-fold (e.g., python calibration.py Im40.txt 10)
# Where the input parameters are
# -> dataset.txt: the name of the file (e.g., Im40.txt, avaialble at http://hyperlabelme.uv.es/)
# -> n-fold: the number of folds for the cross validation 

if __name__ == "__main__":
    
    if len(sys.argv) < 3:
         print('Usage: '+sys.argv[0]+' dataset.txt n-fold')

    main()
    exit(-1)
