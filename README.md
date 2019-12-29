# SVM_Quantum Annealer

Approching Remote Sensing Image Classification with Ensembles of Support Vector Machines on the D-Wave Quantum Annealer


Support  Vector  Machine  (SVM)  is  a  supervised  Machine Learning  (ML)  method  that  is  widely  used  for  the  classification  of  land-cover  and  land-use  
classes  in  Remote  Sensing  (RS)  images. A  method  to  train  SVMs  on  a  D-Wave2000Q  Quantum  Annealer  (QA)  was  recently  proposed  forbinary classification 
problems on biology data. First, ensembles of sub-optimal quantum SVMs are generated by training each  classifier  on  a  disjoint  training  subset  that  can  be  fit into  
the QA. Then, the computed suboptimal solutions are adopted for making predictions on unseen data.   

D. Willsch, M. Willsch, H. De Raedt and K. Michielsen, ''Support Vector Machines on the D-Wave Quantum Annealer'' 2019. 
[Online]. Available:http://dx.doi.org/10.1016/j.cpc.2019.107006


This repository contains the Python functions and the processing pipeline documented in a Jupyter notebook for performing classification of RS images  
with the SVM on the D-Wave2000Q (QA). 

Everyone can make a free account to run on the \ac{DW2000Q} computer: 


- Make a free account to run on the D-Wave through 👉 (https://www.dwavesys.com/take-leap

- Install Ocean Software with 'pip install dwave-ocean-sdk' 👉 https://docs.ocean.dwavesys.com/en/latest/overview/install.html

- Configuring the D-Wave System as a Solver with 'dwave config create' 👉 https://docs.ocean.dwavesys.com/en/latest/overview/dwavesys.html#dwavesys)

