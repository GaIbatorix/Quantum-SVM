# Created by Gabriele Cavallaro (g.cavallaro@fz-juelich.de) 

from quantum_SVM import *
from sklearn import preprocessing


def main():
    
    path_dataset=sys.argv[1] # Path+data (file.txt)
    id_dataset=sys.argv[1][len(sys.argv[1])-8:-4] 
    experiments=int(sys.argv[2])
    
    # These paths should be created in advance 
    train_path='input_datasets/train/'+id_dataset+'/'  # path_data_key
    train_key = id_dataset+'calibtrain'
    train_out='outputs/train/'+id_dataset+'/'
    test_out='outputs/test/'+id_dataset+'/'

    # Load the alphas 
    path_files=np.load(train_out+'trained_SVMs.npy')
    
    # Load the data
    [X_train, Y_train, X_test]=dataread(path_dataset)
    
    # Pre-processing the test spectra
    X_test = preprocessing.scale(X_test)

    # Here we store the scores for the predictions 
    scores=[]

    for j in range(0,experiments):
        for i in range(0,len(path_files)):
            scores.append(predict(train_path,path_files[i],X_test))
 
    avg_scores=np.zeros((scores[0].shape[0]))
    Y_predicted=np.zeros((scores[0].shape[0]),int)

    for i in range(0,scores[0].shape[0]):
        tmp=0
        for y in range(0,slices):
            tmp=tmp+scores[y][i]
        avg_scores[i]=tmp/slices   
 
    # Put back the classes numbers to 1 and 2 (accepted by HyperLabelMe)
    for i in range(0,scores[0].shape[0]):
        if(avg_scores[i]<0):
            Y_predicted[i]=1
        else:
            Y_predicted[i]=2
        

    datawrite(test_out+'SVM', 'Im40', Y_predicted)
    

# In the path where the code is placed, create the following folders
#  -> input_datasets/test/id_dataset (e.g., id_dataset: Im40 )
#  -> outputs/test/id_dataset

# How to run the test
# -> python test.py dataset.txt number_of_experiments (e.g., python test.py Im40.txt 1)
# Where the input parameters are
# -> dataset.txt: the name of the file (e.g., Im40.txt, avaialble at http://hyperlabelme.uv.es/)
# -> number_of_experiments: the number of runs of each test experiment  


if __name__ == "__main__":
    
    if len(sys.argv) < 3:
         print('Usage: '+sys.argv[0]+' dataset.txt number_of_experiments')

    main()
    exit(-1)
