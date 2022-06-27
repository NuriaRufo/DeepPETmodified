import numpy as np

def load_data(Set):
    X_train = []
    Y_train = []
    pathx='/home/nrufo/Final_final//Data_original/set2/Training_data/'+str(Set)+'_setXtrain.npy'
    pathy='/home/nrufo/Final_final//Data_original/set2/Training_data/'+str(Set)+'_setYtrain.npy'
    X_train = np.load(pathx)
    Y_train = np.load(pathy)
    
    noiseFactor=160
    X_train= X_train*noiseFactor
    X_train = np.random.poisson(X_train)   
    return np.array(X_train), np.array(Y_train)

def batch_generator(numberSets, batch_size):
    ids = np.arange(numberSets)
    #batch=[]
    while True:
            np.random.shuffle(ids) 
            i=0
            for i in ids:
                yield load_data(i)
                        #batch=[]