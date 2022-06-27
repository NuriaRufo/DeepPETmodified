
import numpy as np
import astra
import time
from scipy import ndimage


def REC_EM(sinogram,projector_id,iterations):
    
    xsize=sinogram.shape[0]
    ysize=sinogram.shape[1]
    sinogram= np.reshape(sinogram,(xsize*ysize))
    W = astra.OpTomo(projector_id)
    x= np.ones((xsize,ysize))

    n=0
    
    while n<iterations:
        yP= W*x
        yN=sinogram/yP
        xR=W.T*yN
        x= np.reshape(x,(xsize*ysize))
        x=x*xR
        xS = W.T*np.ones((xsize*ysize));
        x=x/xS
        x=np.reshape(x,(xsize,ysize))
        n+=1
    
    x = (x - x.min())/ (x.max() - x.min())
        
    return x

def Rec_FBP(sinogram,proj_id,vol_geom,proj_geom):
    
    
    sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram)
    rec_id = astra.data2d.create('-vol', vol_geom)
    cfg = astra.astra_dict('FBP_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['ProjectorId'] = proj_id
    cfg['option'] = {}
    cfg['option']['FilterType'] = 'hamming'

    alg_id = astra.algorithm.create(cfg)
    #time.sleep(0.00000001)#if not kernel dies
    astra.algorithm.run(alg_id)
    #time.sleep(0.00000001)
    x = astra.data2d.get(rec_id)
    x = (x - x.min())/ (x.max() - x.min())
    
    return x

def REC_EM_REG(sinogram,projector_id,beta, iterations):
    
    xsize=sinogram.shape[0]
    ysize=sinogram.shape[1]
    
    sinogram= np.reshape(sinogram,(xsize*ysize))
    W = astra.OpTomo(projector_id)
    x= np.ones((xsize,ysize))
    n=0
    
    while n<iterations:
        yP= W*x
        p_r=sinogram/yP
        x= np.reshape(x,(xsize*ysize))
        v_norm = W.T*np.ones((xsize*ysize));
        x_r=(W.T*p_r)/v_norm
    
        x_x = np.reshape(x,(xsize,ysize))
        x_median  = ndimage.median_filter(x_x, 3)
        x_median = np.reshape(x_median,(xsize*ysize))
        C =  1 /(1 + beta*(x-x_median)/x_median)#regularization term
    
    
        x=C*(x*x_r)
        x=np.reshape(x,(xsize,ysize))
        n+=1
    x = (x - x.min())/ (x.max() - x.min())
        
    return x

'''
def Rec_FBP_total(Sinograms, proj_geom, vol_geom):
    
    nIm = Sinograms.shape[0]
    xsize = Sinograms.shape[1]
    ysize = Sinograms.shape[2]
    
    recIm_FPB= np.zeros((nIm,xsize,ysize))
    n=0

    tic1 = time.time()

    proj_id = astra.create_projector('linear', proj_geom, vol_geom)
    while n<nIm:
    
    sinogram= np.squeeze(X_test[n,:,:])
    rec= Rec_FBP(sinogram,proj_id,vol_geom,proj_geom)
    recIm_FPB[n,:,:]= rec
    n+=1

    toc1 = time.time()
    Time_FPB= toc1 - tic1
    print('FBP reconstruction in ',Time_FPB,'s' )
    
    return recIm_FBP

def Rec_EM_total(Sinograms, iteration, proj_geom, vol_geom):
    
    nIm = Sinograms.shape[0]
    xsize = Sinograms.shape[1]
    ysize = Sinograms.shape[2]
    
    
    nIm= X_test.shape[0]
    recIm_EM= np.zeros((nIm,xsize,ysize))
    n=0

    projector_id = astra.create_projector('linear', proj_geom, vol_geom)

    tic1 = time.time()
    while n<nIm:
    sinogram= np.squeeze(X_test[n,:,:])
    rec= REC_EM(sinogram,projector_id,iteration)
    recIm_EM[n,:,:]= rec
    n+=1

    toc1 = time.time()
    Time_EM= toc1 - tic1
    print('EM reconstruction in ',Time_EM,'s' )

    
    return recIm_EM


def Rec_EM_total(Sinograms, iteration,beta, proj_geom, vol_geom):
    
    nIm = Sinograms.shape[0]
    xsize = Sinograms.shape[1]
    ysize = Sinograms.shape[2]
    
    
    recIm_EMR= np.zeros((nIm,xsize,ysize))
    n=0
    projector_id = astra.create_projector('linear', proj_geom, vol_geom)

    tic1 = time.time()
    while n<nIm:
    sinogram= np.squeeze(X_test[n,:,:])
    rec= REC_EM_REG(sinogram,projector_id,beta,iteration)
    recIm_EMR[n,:,:]= rec
    n+=1

    toc1 = time.time()
    Time_EMR= toc1 - tic1
    print('EMR reconstruction in ',Time_EMR,'s' )
    
    return recIm_EMR

def BestIteration(Images,Sinograms, method)

    nIm=40 #choose a random one
    Image=np.squeeze(Images[nIm,:,:])
    sinogram=np.squeeze(Sinograms[nIm,:,:])

    tries=60
    start=20
    m=start
    errors=np.zeros(tries-m)
    cont=0
    beta=0.2
    while m<tries:
        iteration=m
        
        if method== EM:
            rec= REC_EM(sinogram,proj_id,iteration) #choose which one to study
            
        else: 
            rec= REC_EM_REG(sinogram,proj_id,0.2,iteration)
        ssim_E= ssim(Image,rec)
        errors[cont]=ssim_E
        m+=1
        cont+=1

    x=errors.max()
    n=0
    while n<errors.size:
        if errors[n]==x:
            print(n)
            opt=n
        n+=1

    print('The best iteration number is '+str(opt-1+start)+' with a SSIM: '+str(errors[opt]))
'''