import numpy as np
import cv2
import numpy as np
import nibabel as nib

from DataAugmentation import Aug_Rotate, Aug_translate, Aug_Zoom


import time
import numpy as np
import scipy.io as sio
from skimage.transform import iradon,radon
from skimage import util
import random
import cv2
import astra


## PERSONALIZE YOUR PARAMETERS#######################################################################
#Path to find your images                                                                           #
                                                                                                    #
                                                                                                    #
path =                                                                                              #
file =                                                                                              #
# imagesfile =  #indicate the name of your new numpy containing images and sinograms                #
# sinogramfile =   #

#Data Augmenation
Rotations = 0
Zoom = 0
Translation = 2
                                                                                                    #
# FINAL PATH TO SAVE YOUR NEW SET                                                                   #
pathfinal =                                                                                         #
                                                                                                    #
                                                                                                    #
#####################################################################################################

# IMPORT YOUR IMAGES
Images = np.load(path+file)

print('Images imported')

nIm = Images.shape[0]
xsize = Images.shape[1]
xsize = Images.shape[2]

# DATA AUGMENTATION

#Images= Aug_Zoom(Images,Zoom)
#Images= Aug_Rotate(Images,Rotations)
#Images= Aug_translate(Images,Translation)

# GENERATE SINOGRAMS
# Create geometries and projector.

print('Generating sinograms...')

vol_geom = astra.create_vol_geom(xsize, ysize)
angles = np.linspace(0, np.pi, xsize, endpoint=False)
proj_geom = astra.create_proj_geom('parallel', 1., xsize, angles)
projector_id = astra.create_projector('linear', proj_geom, vol_geom)
 
# Create sinogram.
nSin = Images.shape[0]
Sinograms = np.zeros((nSin,xsize,ysize,1))
n=0
cont=0
while n<nSin:
    im= np.squeeze(Images[n,:,:,0])
    sinogram_id, sinogram = astra.create_sino(im, projector_id)
    Sinograms[cont,:,:,0]=sinogram
    cont+=1
    n+=1
    
Sinograms = (Sinograms - Sinograms.min())/ (Sinograms.max() - Sinograms.min())

print('Sinograms generated correctly!')
      
# SAVE YOUR DATA
    
np.save(pathfinal+imagesfile,Images) 
np.save(pathfinal+sinogramfile,Sinograms)
'''
# SAVE YOUR DATA IN BATCHES

ids= Sinograms.shape[0]
bs=32

batch=np.zeros((bs,xsize,ysize,1))
batch1=np.zeros((bs,xsize,ysize,1))
n=0
cont=0
cont3=0
cont2=0
while n<ids:
    batch[cont,:,:,:]=X_train[n,:,:,:]
    batch1[cont,:,:,:]=Y_train[n,:,:,:]
    cont+=1
    if cont==bs-1:
        #p='/home/nrufo/Final_final/Data_original/set1/Training_data/'
        path = pathfinal +str(cont2)+'_setXtrain.npy'
        path1 = pathfinal +str(cont2)+'_setYtrain.npy'
        np.save(path,batch)
        np.save(path1,batch1)
        cont=0
        cont2+=1
        cont3+=1
    if cont3+2> ids/bs:
        n=ids+100
    n+=1
'''

# DATA AUGMENTATION METHODS

def Aug_Rotate(aug, augNumber):
    
    
     # aug= what I will give as parameter
    # augNumber= how many new images i will perform for each image
    Nim= aug.shape[0]
    xsize=aug.shape[1]
    ysize=aug.shape[2]
    
    tic1 = time.time()
    print ('Augmenting data ... ')

    newAug=np.zeros((Nim*augNumber,aug.shape[1],aug.shape[2],aug.shape[3]))
    
    cont=0 # to save new images in the new matrix, final size = augnumber*Nim                
    n=0
   
    while n<Nim:# for each image
        augIm=np.squeeze(aug[n,:,:])
        newAug[cont,:,:,0]=augIm# we keep the original image
        cont+=1
        a=0
        while a<augNumber-1:# number of new images
            v=augIm[2,2]
            Rotation=random.randint(0,360)
            M = cv2.getRotationMatrix2D((xsize//2,ysize//2),Rotation,1)
            AugImPos = cv2.warpAffine(augIm,M,(xsize,ysize)) # apply translation
            x=0
            black=0.041218829656755294 #revise to write an elegant way
            while x<xsize:# to avoid false zeros
                y=0
                while y<ysize:
                    if  AugImPos[x,y] <=v+0.0001:
                        AugImPos[x,y]=0
       
                    y+=1
                x+=1  
            # we save the new images 
            
            newAug[cont,:,:,0]= AugImPos
            cont+=1
            a+=1
                
        #Normalize the result      
        n+=1
                    
    newAug=(newAug - newAug.min())/ (newAug.max() - newAug.min())     
    toc1 = time.time()
    print('Augmentation done in ',(toc1 - tic1),' !' )
    
    return newAug

def Aug_translate(aug, augNumber):
    import random
    
     # aug= what I will give as parameter
    # augNumber= how many new images i will perform for each image
    Nim= aug.shape[0]
    xsize=aug.shape[1]
    ysize=aug.shape[2]
    
    tic1 = time.time()
    print ('Augmenting data ... ')

    newAug=np.zeros((Nim*augNumber,aug.shape[1],aug.shape[2],aug.shape[3]))
    
    cont=0 # to save new images in the new matrix, final size = augnumber*Nim                
    n=0
    
    while n<Nim:# for each image
        augIm=np.squeeze(aug[n,:,:])
        newAug[cont,:,:,0]=augIm# we keep the original image
        cont+=1
        a=0
        while a<augNumber-1:# number of new images
        
            TrasX= random.randint(-15,15)
            ShearX=0
            ShearY=0
            TrasY=random.randint(-15,15)
            M = np.float32([[1,ShearX, TrasX],[ShearY,1, TrasY]])# define the translation
    
            AugImPos = cv2.warpAffine(augIm,M,(xsize,ysize)) # apply translation
            x=0
            black=augIm[2,2]#revise to write an elegant way
            while x<xsize:# to avoid false zeros
                y=0
                while y<ysize:
                    if  AugImPos[x,y] <=black:
                        AugImPos[x,y]=0
       
                    y+=1
                x+=1  
            # we save the new images 
            
            newAug[cont,:,:,0]= AugImPos
            cont+=1
            a+=1
                
           
        n+=1
        #Normalize the result               
    newAug=(newAug - newAug.min())/ (newAug.max() - newAug.min())     
    toc1 = time.time()
    print('Augmentation done in ',(toc1 - tic1),' !' )
    
    return newAug