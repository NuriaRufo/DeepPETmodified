import time
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from skimage.transform import iradon
from skimage import util
import cv2
import random
import tensorflow as tf

def createCircularMask(h, w, center=None, radius=None):
    #draw a circular mask
    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def crop_circle(image, crop_half_width, crop_half_height, mask):
    #crop the center of an image and draw a circular mask '   
    h=image.shape[1]
    w=image.shape[0]
    center = [int(w/2), int(h/2)]
    masked_img = image[int(center[0]-crop_half_width):int(center[0]+crop_half_width),int(center[1]-crop_half_height):int(center[1]+crop_half_height)].copy()

    masked_img[~mask] = 0

    return (masked_img)


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

def Aug_RotateAlterno(aug, augNumber):
    
    
     # aug= what I will give as parameter
    # augNumber= how many new images i will perform for each image
    Nim= aug.shape[0]

    tic1 = time.time()
    print ('Augmenting data ... ')

    newAug=np.zeros((Nim,aug.shape[1],aug.shape[2],aug.shape[3]))
    
    cont=0 # to save new images in the new matrix, final size = augnumber*Nim                
    n=0
    j=0
    while n<Nim:# for each image
       
        augIm=np.squeeze(aug[n,:,:])
        
        if j==0:
            newAug[cont,:,:,0]=augIm# we keep the original image
            cont+=1
            j=1
            
        else:
            v=augIm[2,2]
            Rotation=random.randint(0,360)
            M = cv2.getRotationMatrix2D((256//2,256//2),Rotation,1)
            AugImPos = cv2.warpAffine(augIm,M,(256,256)) # apply translation
            x=0
            while x<256:# to avoid false zeros
                y=0
                while y<256:
                    if  AugImPos[x,y] <=v+0.0001:
                        AugImPos[x,y]=0
       
                    y+=1
                x+=1
            newAug[cont,:,:,0]= AugImPos
            cont+=1
            j=0
            
        n+=1
          
                    
    newAug=(newAug - newAug.min())/ (newAug.max() - newAug.min())     
    toc1 = time.time()
    print('Augmentation done in ',(toc1 - tic1),' !' )
    
    return newAug

def Aug_RotateNoOriginal(aug, augNumber):
    
    
     # aug= what I will give as parameter
    # augNumber= how many new images i will perform for each image
    Nim= aug.shape[0]

    tic1 = time.time()
    print ('Augmenting data ... ')

    newAug=np.zeros(((Nim*augNumber)-Nim,aug.shape[1],aug.shape[2],aug.shape[3]))
    
    cont=0 # to save new images in the new matrix, final size = augnumber*Nim                
    n=0
    while n<Nim:# for each image
        augIm=np.squeeze(aug[n,:,:])
        #newAug[cont,:,:,0]=augIm# we keep the original image
        #cont+=1
        a=0
        while a<augNumber-1:# number of new images
            v=augIm[2,2]
            Rotation=random.randint(0,360)
            M = cv2.getRotationMatrix2D((256//2,256//2),Rotation,1)
            AugImPos = cv2.warpAffine(augIm,M,(256,256)) # apply translation
            x=0
            black=0.041218829656755294 #revise to write an elegant way
            while x<256:# to avoid false zeros
                y=0
                while y<256:
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

def Aug_Zoom(aug, augNumber):
    
    
     # aug= what I will give as parameter
    # augNumber= how many new images i will perform for each image
    Nim= aug.shape[0]

    tic1 = time.time()
    print ('Augmenting data ... ')

    newAug=np.zeros((Nim*augNumber,aug.shape[1],aug.shape[2],aug.shape[3]))
    # we keep the original image
    cont=0 # to save new images in the new matrix, final size = augnumber*Nim                
    n=0
    while n<Nim:# for each image
        augIm=np.squeeze(aug[n,:,:])
        newAug[cont,:,:,0]=augIm# we keep the original image
        cont+=1
        a=0
        while a<augNumber-1:# number of new images
        
            Zoom=random.randint(80,130) # zoom from 0.8 to 1.3
            Zoom/=100
            M = cv2.getRotationMatrix2D((256//2,256//2),0,Zoom)
            AugImPos = cv2.warpAffine(augIm,M,(256,256)) # apply translation
            x=0
            black=augIm[0,0] #revise to write an elegant way
            while x<255:# to avoid false zeros
                y=0
                while y<255:
                    if  AugImPos[x,y] <black+0.0000000001:
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