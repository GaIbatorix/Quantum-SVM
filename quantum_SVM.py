# Created by Dennis Willsch (d.willsch@fz-juelich.de) 
# Modified by Gabriele Cavallaro (g.cavallaro@fz-juelich.de) 

import os
import sys
import re
import numpy as np
import numpy.lib.recfunctions as rfn
import matplotlib.pyplot as plt
from utils import *

import shutil
import pickle
import numpy.lib.recfunctions as rfn
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dimod import BinaryQuadraticModel



def gen_svm_qubos(B,K,xi,gamma,path_data_key,data_key,path_out):
         
    data,label = loaddataset(path_data_key+data_key)

    N = len(data)

    Q = np.zeros((K*N,K*N))
    print(f'Creating the QUBO of size {Q.shape}')
    for n in range(N): # not optimized: size will not be so large and this way its more easily verifyable
        for m in range(N):
            for k in range(K):
                for j in range(K):
                    Q[K*n+k,K*m+j] = .5 * B**(k+j) * label[n] * label[m] * (kernel(data[n], data[m], gamma) + xi)
                    if n == m and k == j:
                        Q[K*n+k,K*m+j] += - B**k


    print(f'Extracting nodes and couplers')
    Q = np.triu(Q) + np.tril(Q,-1).T # turn the symmetric matrix into upper triangular
    qubo_nodes = np.asarray([[n, n, Q[n,n]] for n in range(len(Q))])  # if not np.isclose(Q[n,n],0)]) NOTE: removed due to variable order!
    qubo_couplers = np.asarray([[n, m, Q[n,m]] for n in range(len(Q)) for m in range(n+1,len(Q)) if not np.isclose(Q[n,m],0)])
    qubo_couplers = qubo_couplers[np.argsort(-np.abs(qubo_couplers[:,2]))]

    #path = f'runs/run{data_key}_B={B}_K={K}_xi={xi}_gamma={gamma}/'   
    path = f'{path_out}run{data_key}_B={B}_K={K}_xi={xi}_gamma={gamma}/'
    print(f'Saving {len(qubo_nodes)} nodes and {len(qubo_couplers)} couplers for {path}')
    os.makedirs(path, exist_ok=True)
    np.save(path+'Q.npy', Q)
    np.savetxt(path+'qubo_nodes.dat', qubo_nodes, fmt='%g', delimiter='\t')
    np.savetxt(path+'qubo_couplers.dat', qubo_couplers, fmt='%g', delimiter='\t')
    
    return path


def dwave_run(path_data_key,path_in):
    
    MAXRESULTS = 20 # NOTE: to save space only 20 best results
    match = re.search('run([^/]*)_B=(.*)_K=(.*)_xi=(.*)_gamma=([^/]*)', path_in) 

    data_key = match.group(1)
    B = int(match.group(2))
    K = int(match.group(3))
    xi = float(match.group(4))
    gamma = float(match.group(5))   
    data,label = loaddataset(path_data_key+data_key)

    path = path_in+ ('/' if path_in[-1] != '/' else '')
    qubo_couplers = np.loadtxt(path+'qubo_couplers.dat')
    qubo_nodes = np.loadtxt(path+'qubo_nodes.dat')
    qubo_nodes = np.array([[i,i,(qubo_nodes[qubo_nodes[:,0]==i,2][0] if i in qubo_nodes[:,0] else 0.)] for i in np.arange(np.concatenate((qubo_nodes,qubo_couplers))[:,[0,1]].max()+1)])  # to make sure every (i,i) occurs in the qubo in increasing order such that the variable order in BinaryQuadraticModel is consistent (see locate wrongenergies-* github issue)
    maxcouplers = len(qubo_couplers) ## POSSIBLE INPUT if len(sys.argv) <= 2 else int(sys.argv[2])

    if not 'train' in data_key:
        raise Exception(f'careful: datakey={data_key} => youre trying to train on a validation / test set!')

    couplerslist = [maxcouplers]
    for trycouplers in [2500, 2000, 1800, 1600, 1400, 1200, 1000, 500]:
        if maxcouplers > trycouplers:
            couplerslist += [trycouplers]

    sampler = EmbeddingComposite(DWaveSampler())
    for couplers in couplerslist:  # try to reduce as little couplers as necessary to find an embedding
        Q = { (q[0], q[1]): q[2] for q in np.vstack((qubo_nodes, qubo_couplers[:couplers])) }
        pathsub = path + f'result_couplers={couplers}/'
        os.makedirs(pathsub, exist_ok=True)
        print(f'running {pathsub} with {len(qubo_nodes)} nodes and {couplers} couplers')
    
        ordering = np.array(list(BinaryQuadraticModel.from_qubo(Q)))
        if not (ordering == np.arange(len(ordering),dtype=ordering.dtype)).all():
            print(f'WARNING: variables are not correctly ordered! path={path} ordering={ordering}')

        try:
            response = sampler.sample_qubo(Q, num_reads=10000)  # NOTE: if the scale of the Qij is very different from 1, one should not use
                                                                # the default chain_strength=1 for the embedding here because the
                                                                # embedding would not use properly scaled strengths to tie physical qubits together
                                                                # (This will show up in a large chain_break_fraction)
                                                                # Instead, a great approach is to set
                                                                #   chain_strength = r * max(abs(Qij))
                                                                # for r = 1.0, 0.9, 0.8, ... until the best chain_strength is found.
        except ValueError as v:
            print(f' -- no embedding found, removing {pathsub} and trying less couplers')
            shutil.rmtree(pathsub)
            continue
        break

    save_json(pathsub+'info.json', response.info) # contains response.info
    #NOTE left out: pickle.dump(response, open(pathsub+'response.pkl','wb')) # contains full response incl. response.record etc; can be loaded with pickle.load(open('response.pkl','rb'))

    samples = np.array([''.join(map(str,sample)) for sample in response.record['sample']]) # NOTE: it would be safer to use the labeling from record.data() for the qubit variable order
    unique_samples, unique_idx, unique_counts = np.unique(samples, return_index=True, return_counts=True) # unfortunately, num_occurrences seems not to be added up after unembedding
    unique_records = response.record[unique_idx]
    result = rfn.merge_arrays((unique_samples, unique_records['energy'], unique_counts, unique_records['chain_break_fraction']))  # see comment on chain_strength above
    result = result[np.argsort(result['f1'])]
    np.savetxt(pathsub+'result.dat', result[:MAXRESULTS], fmt='%s', delimiter='\t', header='\t'.join(response.record.dtype.names), comments='') # load with np.genfromtxt(..., dtype=['<U2000',float,int,float], names=True, encoding=None)

    alphas = np.array([decode(sample,B,K) for sample in result['f0'][:MAXRESULTS]])
    np.save(pathsub+f'alphas{data_key}_gamma={gamma}.npy', alphas)
    
    return pathsub


def eval_run_trainaccuracy(path_in):
    

    regex = 'run([^/]*)_B=(.*)_K=(.*)_xi=(.*)_gamma=([^/]*)/result_couplers.*/?$'
    match = re.search(regex, path_in)
 
    path = path_in + ('/' if path_in[-1] != '/' else '')
    data_key = match.group(1)
    B = int(match.group(2))
    K = int(match.group(3))
    xi = float(match.group(4))
    gamma = float(match.group(5))
    data,label = loaddataset(data_key)

    alphas_file = path+f'alphas{data_key}_gamma={gamma}.npy'
    if not os.path.isfile(alphas_file):
        print('result '+alphas_file+' doesnt exist, exiting')
        sys.exit(-1)

    alphas = np.atleast_2d(np.load(alphas_file))
    nalphas = len(alphas)
    assert len(data) == alphas.shape[1], "alphas do not seem to be for the right data set?)"

    result = np.genfromtxt(path+'result.dat', dtype=['<U2000',float,int,float], names=True, encoding=None, max_rows=nalphas)

    Cs = [100, 10, (B**np.arange(K)).sum(), 1.5]
    evaluation = np.zeros(nalphas, dtype=[('sum_antn',float)]+[(f'acc(C={C})',float) for C in Cs])

    for n,alphas_n in enumerate(alphas):
        evaluation[n]['sum_antn'] = (label * alphas_n).sum()
        for j,field in enumerate(evaluation.dtype.names[1:]):
            b = eval_offset_avg(alphas_n, data, label, gamma, Cs[j]) # NOTE: this is NAN if no support vectors were found, see TODO file
            label_predicted = np.sign(eval_classifier(data, alphas_n, data, label, gamma, b)) # NOTE: this is only train accuracy! (see eval_result_roc*)
            evaluation[n][field] = (label == label_predicted).sum() / len(label)

    result_evaluated = rfn.merge_arrays((result,evaluation), flatten=True)
    fmt = '%s\t%.3f\t%d\t%.3f' + '\t%.3f'*len(evaluation.dtype.names)
    #NOTE: left out
    # np.savetxt(path+'result_evaluated.dat', result_evaluated, fmt=fmt, delimiter='\t', header='\t'.join(result_evaluated.dtype.names), comments='') # load with np.genfromtxt(..., dtype=['<U2000',float,int,float,float,float,float,float], names=True, encoding=None)

    print(result_evaluated.dtype.names)
    print(result_evaluated)
    
    

def eval_run_rocpr_curves(path_data_key,path_in,plotoption):  

    regex = 'run([^/]*)_B=(.*)_K=(.*)_xi=(.*)_gamma=([^/]*)/result_couplers.*/?$' 
    match = re.search(regex, path_in) 

    path = path_in + ('/' if path_in[-1] != '/' else '')
    data_key = match.group(1)
    B = int(match.group(2))
    K = int(match.group(3))
    xi = float(match.group(4))
    gamma = float(match.group(5))
    data,label = loaddataset(path_data_key+data_key)
  
    dwavesolutionidx=0
    C=(B**np.arange(K)).sum()

    if 'calibtrain' in data_key:
        testname = 'Validation'
        datatest,labeltest = loaddataset(path_data_key+data_key.replace('calibtrain','calibval'))
    else:
        print('be careful: this does not use the aggregated bagging classifier but only the simple one as in calibration')
        testname = 'Test'
        datatest,labeltest = loaddataset(re.sub('train(?:set)?[0-9]*(?:bag)[0-9]*','test',data_key))

    alphas_file = path+f'alphas{data_key}_gamma={gamma}.npy'
    if not os.path.isfile(alphas_file):
        print('result '+alphas_file+' doesnt exist, exiting')
        sys.exit(-1)

    alphas = np.atleast_2d(np.load(alphas_file))
    nalphas = len(alphas)
    assert len(data) == alphas.shape[1], "alphas do not seem to be for the right data set?)"

    print('idx   \tsum_antn\ttrainacc\ttrainauroc\ttrainauprc\ttestacc  \ttestauroc\ttestauprc')
    
    trainacc_all=np.zeros([nalphas])
    trainauroc_all=np.zeros([nalphas])
    trainauprc_all=np.zeros([nalphas])
    
    testacc_all=np.zeros([nalphas])
    testauroc_all=np.zeros([nalphas])
    testauprc_all=np.zeros([nalphas])

    for i in range(nalphas):
        alphas_n = alphas[i]
        b = eval_offset_avg(alphas_n, data, label, gamma, C) # NOTE: this is NAN if no support vectors were found, see TODO file
        score = eval_classifier(data, alphas_n, data, label, gamma, b)
        scoretest = eval_classifier(datatest, alphas_n, data, label, gamma, b)
        trainacc,trainauroc,trainauprc = eval_acc_auroc_auprc(label,score)
        testacc,testauroc,testauprc = eval_acc_auroc_auprc(labeltest,scoretest)
        
        trainacc_all[i]=trainacc
        trainauroc_all[i]=trainauroc
        trainauprc_all[i]=trainauprc
        testacc_all[i]=testacc
        testauroc_all[i]=testauroc
        testauprc_all[i]=testauprc

        print(f'{i}\t{(label*alphas_n).sum():8.4f}\t{trainacc:8.4f}\t{trainauroc:8.4f}\t{trainauprc:8.4f}\t{testacc:8.4f}\t{testauroc:8.4f}\t{testauprc:8.4f}')
                   

    # plot code starts here
    if plotoption != 'noplotsave':
        alphas_n = alphas[dwavesolutionidx] # plot only the requested
        b = eval_offset_avg(alphas_n, data, label, gamma, C) # NOTE: this is NAN if no support vectors were found, see TODO file
        score = eval_classifier(data, alphas_n, data, label, gamma, b)
        scoretest = eval_classifier(datatest, alphas_n, data, label, gamma, b)

        # roc curve
        plt.figure(figsize=(6.4,3.2))
        plt.subplot(1,2,1)
        plt.subplots_adjust(top=.95, right=.95, bottom=.15, wspace=.3)
        fpr, tpr, thresholds = roc_curve(labeltest, scoretest)
        auroc = roc_auc_score(labeltest, scoretest)
        plt.plot(fpr, tpr, label='AUROC=%0.3f' % auroc, color='g')
        plt.fill_between(fpr, tpr, alpha=0.2, color='g', step='post')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        #plt.title('Receiver Operating Curve')
        plt.legend(loc="lower right")
        # pr curve
        plt.subplot(1,2,2)
        precision, recall, _ = precision_recall_curve(labeltest, scoretest)
        auprc = auc(recall, precision)
        plt.step(recall, precision, color='g', where='post',
            label='AUPRC=%0.3f' % auprc)
        plt.fill_between(recall, precision, alpha=0.2, color='g', step='post')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.0])
        plt.xlim([0.0, 1.0])
        #plt.title('PR curve')
        plt.legend(loc="lower right")

        # save the data for gnuplot
        savename = f'{path.replace("/","_")}{dwavesolutionidx}'
        #with open('results/rocpr_curves/'+savename,'w') as out:
        with open(path_in+savename,'w') as out:
            out.write(f'AUROC\t{auroc:0.3f}\t# ROC:FPR,TPR\n')
            assert len(fpr) == len(tpr)
            for i in range(len(fpr)):
                out.write(f'{fpr[i]}\t{tpr[i]}\n')
            out.write(f'\n\nAUPRC\t{auprc:0.3f}\t# PRC:Recall,Precision\n')
            assert len(recall) == len(precision)
            for i in range(len(recall)):
                out.write(f'{recall[i]}\t{precision[i]}\n')
            print(f'saved data for {savename}')

        if plotoption == 'saveplot':
            savefigname = path_in+savename+'.pdf'
            plt.savefig(savefigname)
            print(f'saved as {savefigname}')
        else:
            plt.show()
            
    
    return np.average(trainacc_all), np.average(trainauroc_all), np.average(trainauprc_all) ,np.average(testacc_all), np.average(testauroc_all), np.average(testauprc_all)
            
    
def predict(path_data_key,path_in,datatest):  

    regex = 'run([^/]*)_B=(.*)_K=(.*)_xi=(.*)_gamma=([^/]*)/result_couplers.*/?$'  
    match = re.search(regex, path_in) 

    path = path_in + ('/' if path_in[-1] != '/' else '')
    data_key = match.group(1)
    B = int(match.group(2))
    K = int(match.group(3))
    xi = float(match.group(4))
    gamma = float(match.group(5))
    data,label = loaddataset(path_data_key+data_key)
    
    C=(B**np.arange(K)).sum()
    
    # Load the alphas (20xnumber of samples)
    #alphas=np.load(path_files[y]+f'alphas{data_key}{i}_{y}_gamma={gamma}.npy')
    alphas = np.atleast_2d(np.load(path_in+f'alphas{data_key}_gamma={gamma}.npy'))
    nalphas = len(alphas)
    #print(nalphas)
    
    # Compute the mean of the alphas 
    alphas_avg=np.mean(alphas,axis=0)
    
    b = eval_offset_avg(alphas_avg, data, label, gamma, C) # NOTE: this is NAN if no support vectors were found, see TODO file
    
    scoretest = eval_classifier(datatest, alphas_avg, data, label, gamma, b)
      
    return scoretest
