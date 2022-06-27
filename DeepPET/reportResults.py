
import numpy as np
import matplotlib.pyplot as plt
import os

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_root_mse as nrmse

def average_error(ground_truth, rec_FBP, rec_EM, rec_EMR, rec_Deep):
    ssim_fbp =np.zeros(ground_truth.shape[0])
    ssim_EM = np.zeros(ground_truth.shape[0])
    ssim_EMR =np.zeros(ground_truth.shape[0])
    ssim_Deep = np.zeros(ground_truth.shape[0])
    
    psnr_fbp = np.zeros(ground_truth.shape[0])
    psnr_EM = np.zeros(ground_truth.shape[0])
    psnr_EMR = np.zeros(ground_truth.shape[0])
    psnr_Deep = np.zeros(ground_truth.shape[0])
    
    nrmse_fbp = np.zeros(ground_truth.shape[0])
    nrmse_EM = np.zeros(ground_truth.shape[0])
    nrmse_EMR = np.zeros(ground_truth.shape[0])
    nrmse_Deep = np.zeros(ground_truth.shape[0])
    

    nIm = ground_truth.shape[0]
    n = 0
    #counters to see how many times deepPET beats tradicional methods
    count_ssim_f, count_ssim_m, count_ssim_mr = [0]*3 
    count_psnr_f, count_psnr_m, count_psnr_mr = [0]*3 
    count_nrsme_f, count_nrsme_m, count_nrsme_mr = [0]*3 
  

    while n<nIm:
        
        Image = np.squeeze(ground_truth[n,:,:])
        sum1=np.sum(np.sum(Image))
        rec_d = np.squeeze(rec_Deep[n,:,:])
        rec_f = rec_FBP[n,:,:]
        rec_m = rec_EM[n,:,:]
        rec_mr = rec_EMR[n,:,:]
        
        sum2=np.sum(np.sum(rec_m))
        sum3=np.sum(np.sum(rec_f))
        sum4=np.sum(np.sum(rec_mr))
        sum5=np.sum(np.sum(rec_d))

        rec_m= rec_m*(sum1/sum2)
        rec_mr= rec_mr*(sum1/sum4)
        rec_f= rec_f*(sum1/sum3)
        rec_d= rec_d*(sum1/sum5)


        
        #Calculate SSIM errors for all reconstructions
        
        ssim_d = ssim(Image,rec_d)
    
        ssim_f = ssim(Image,rec_f)
        ssim_m = ssim(Image,rec_m)
        ssim_mr = ssim(Image,rec_mr)
        
      
        ssim_Deep[n]=(ssim_d)
        ssim_fbp[n]=(ssim_f)
        ssim_EM[n]= (ssim_m)
        ssim_EMR[n] = (ssim_mr)
        
        #Calculate PSNR errors for all reconstructions
        
        psnr_d = psnr(Image,rec_d)
        
        psnr_f = psnr(Image,rec_f)
        psnr_m = psnr(Image,rec_m)
        psnr_mr = psnr(Image,rec_mr)
        
        psnr_Deep[n]=(psnr_d)
        psnr_fbp[n]=(psnr_f)
        psnr_EM[n]=(psnr_m)
        psnr_EMR[n]=(psnr_mr)
        
        #Calculate PSNR errors for all reconstructions

        nrmse_d = nrmse(Image,rec_d)
        nrmse_f = nrmse(Image,rec_f)
        nrmse_m = nrmse(Image,rec_m)
        nrmse_mr = nrmse(Image,rec_mr)
        
        nrmse_Deep[n]=(nrmse_d)
        nrmse_fbp[n]=(nrmse_f)
        nrmse_EM[n]=(nrmse_m)
        nrmse_EMR[n]=(nrmse_mr)
        
        if ssim_d > ssim_f:
            count_ssim_f += 1
        if ssim_d > ssim_m:
             count_ssim_m += 1
        if ssim_d > ssim_mr:
             count_ssim_mr += 1
                
        if psnr_d > psnr_f:
            count_psnr_f += 1
        if psnr_d > psnr_m:
             count_psnr_m += 1
        if psnr_d > psnr_mr:
             count_psnr_mr += 1
                
        if nrmse_d < nrmse_f:
            count_nrsme_f += 1
        if nrmse_d < nrmse_m:
            count_nrsme_m += 1
        if nrmse_d < nrmse_mr:
             count_nrsme_mr += 1
            
        
       
        n+=1
   
    #Calculate average error

    sdt_ssim_Deep = np.std(ssim_Deep)
    sdt_psnr_Deep =np.std(psnr_Deep)
    sdt_nrmse_Deep = np.std(nrmse_Deep)
    
    sdt_ssim_FBP = np.std(ssim_fbp)
    sdt_psnr_FBP = np.std(psnr_fbp)
    sdt_nrmse_FBP = np.std(nrmse_fbp)
    
    sdt_ssim_EM =np.std(ssim_EM)
    sdt_psnr_EM = np.std(psnr_EM)
    sdt_nrmse_EM = np.std(nrmse_EM)
    
    sdt_ssim_EMR =np.std(ssim_EMR)
    sdt_psnr_EMR = np.std(psnr_EMR)
    sdt_nrmse_EMR = np.std(nrmse_EMR)
    
    average_ssim_Deep = (sum(ssim_Deep))/ nIm
    average_psnr_Deep = (sum(psnr_Deep))/ nIm
    average_nrmse_Deep = (sum(nrmse_Deep))/ nIm
    
    average_ssim_FBP = (sum(ssim_fbp))/ nIm
    average_psnr_FBP = (sum(psnr_fbp))/ nIm
    average_nrmse_FBP = (sum(nrmse_fbp))/ nIm
    
    average_ssim_EM = (sum(ssim_EM))/ nIm
    average_psnr_EM = (sum(psnr_EM))/ nIm
    average_nrmse_EM = (sum(nrmse_EM))/ nIm
    
    average_ssim_EMR = (sum(ssim_EMR))/ nIm
    average_psnr_EMR = (sum(psnr_EMR))/ nIm
    average_nrmse_EMR = (sum(nrmse_EMR))/ nIm
    
    
    Difference_f_ssim = ((average_ssim_Deep-average_ssim_FBP)*100)/average_ssim_Deep
    Difference_f_psnr = ((average_psnr_Deep-average_psnr_FBP)*100)/average_psnr_Deep
    Difference_f_nrsme = ((average_nrmse_Deep-average_nrmse_FBP)*100)/average_nrmse_Deep
    
    Difference_m_ssim = ((average_ssim_Deep-average_ssim_EM)*100)/average_ssim_Deep
    Difference_m_psnr = ((average_psnr_Deep-average_psnr_EM)*100)/average_psnr_Deep
    Difference_m_nrsme = ((average_nrmse_Deep-average_nrmse_EM)*100)/average_nrmse_Deep
    
    Difference_mr_ssim = ((average_ssim_Deep-average_ssim_EMR)*100)/average_ssim_Deep
    Difference_mr_psnr = ((average_psnr_Deep-average_psnr_EMR)*100)/average_psnr_Deep
    Difference_mr_nrsme = ((average_nrmse_Deep-average_nrmse_EMR)*100)/average_nrmse_Deep
    
    
    # CALCULATE PORCENTAGE OF SUCCES
    
    
    # porcentage of deep overcoming fbp
    successVSfbp_ssim =  (count_ssim_f /nIm) *100
    successVSfbp_psnr =  (count_psnr_f* 100)/nIm
    successVSfbp_nrsme =  (count_nrsme_f* 100)/nIm
    
    
    # porcentage of deep overcoming EM
    
    successVSEM_ssim =  (count_ssim_m*100)/nIm
    successVSEM_psnr =  (count_psnr_m*100)/nIm
    successVSEM_nrsme =  (count_nrsme_m*100)/nIm
    
    # porcentage of deep overcoming EMR
    
    successVSEMR_ssim =  (count_ssim_mr*100)/nIm
    successVSEMR_psnr =  (count_psnr_mr*100)/nIm
    successVSEMR_nrsme =  (count_nrsme_mr*100)/nIm
    
    # Print results
    
    if Difference_f_ssim>0:
        a = 'higher'
    else:
        a= 'lower'
        Difference_f_ssim = ((average_ssim_Deep-average_ssim_FBP)*100)/average_ssim_FBP
    if Difference_m_ssim>0:
        b = 'higher'
    else:
        b= 'lower'
        Difference_m_ssim = ((average_ssim_Deep-average_ssim_FBP)*100)/average_ssim_EM
    if Difference_mr_ssim>0:
        c = 'higher'
    else:
        c= 'lower'
        Difference_mr_ssim = ((average_ssim_Deep-average_ssim_FBP)*100)/average_ssim_EMR
        
    if Difference_f_psnr>0:
        d = 'higher'
    else: 
        d= 'lower'
        Difference_f_psnr = ((average_psnr_Deep-average_psnr_FBP)*100)/average_psnr_FBP
    if Difference_m_psnr>0:
        e = 'higher'
    else:
        e= 'lower'
        Difference_m_psnr = ((average_psnr_Deep-average_psnr_FBP)*100)/average_psnr_EM
    if Difference_mr_psnr>0:
        f = 'higher'
    else:
        f= 'lower'
        Difference_mr_psnr = ((average_psnr_Deep-average_psnr_FBP)*100)/average_psnr_EMR
        
    if Difference_f_nrsme>0:
        g = 'higher'
    else:
        g= 'lower'
        Difference_f_nrsme = ((average_nrmse_Deep-average_nrmse_FBP)*100)/average_nrmse_FBP
    if Difference_m_nrsme>0:
        h = 'higher'
    else:
        h= 'lower'
        Difference_m_nrsme = ((average_nrmse_Deep-average_nrmse_FBP)*100)/average_nrmse_EM
    if Difference_mr_nrsme>0:
        i = 'higher'
    else:
        i= 'lower'
        Difference_mr_nrsme = ((average_nrmse_Deep-average_nrmse_FBP)*100)/average_nrmse_Deep
    
    partdeep = 'The average error for DeepPET reconstruction is: \n    SSIM = '+str(average_ssim_Deep)+'\n    PSNR = '+str(average_psnr_Deep)+'\n    nRMSE = '+ str(average_nrmse_Deep)
    
    partfbp = '\n \n The average error for FBP reconstruction is: \n    SSIM = '+str(average_ssim_FBP)+'\n    PSNR = '+str(average_psnr_FBP)+'\n    nRMSE = '+ str(average_nrmse_FBP)
    
    partEM = '\n \n The average error for ML-EM reconstruction is: \n    SSIM = '+str(average_ssim_EM)+'\n    PSNR = '+str(average_psnr_EM)+'\n    nRMSE = '+ str(average_nrmse_EM)
    
    partEMR = '\n \n The average error for ML-EM Reg reconstruction is: \n    SSIM = '+str(average_ssim_EMR)+'\n    PSNR = '+str(average_psnr_EMR)+'\n    nRMSE = '+ str(average_nrmse_EMR)
    '''
    overall= '\n \n Therefore, the SSIM in DeePPET was: \n    - '+str(round(abs(Difference_f_ssim)))+' % '+a+' compared with FBP \n    - '+str(round(abs(Difference_m_ssim)))+' % '+b+' compared with EM \n    - '+str(round(abs(Difference_mr_ssim)))+' % '+a+' compared with EMR .\n'+'The PSNR was:  \n    - '+str(round(abs(Difference_f_psnr)))+'% '+d+' compared with FBP \n    - '+str(round(abs(Difference_m_psnr)))+' % '+e+' compared with EM \n    - '+str(round(abs(Difference_mr_psnr)))+' % '+f+' compared with EMR .\n'+'The nRMSE was:  \n    - '+str(round(abs(Difference_f_nrsme)))+' % '+g+' compared with FBP \n    - '+str(round(abs(Difference_m_nrsme)))+' % '+h+' compared with EM \n    - '+str(round(abs(Difference_mr_nrsme)))+' % '+i+' compared with EMR .\n'
   '''
    final=' Finally, deepPET recorded a better performance in the test regarding nRMSE:'
    
    print(partdeep+partfbp+partEM+partEMR)
    
    # REPORT RESULTS
    
    
    
    # PLOT RESULTS
    
    from matplotlib import cm
    from matplotlib import colors
    
    plt.title(" DeepPET VS.  ")        
    plt.subplot(311)
    success_Rate = [successVSfbp_nrsme, 100-successVSfbp_nrsme]
    nombres = ["% of better image reconstruction     \ndeepPET Vs. FBP                 ",""]
    colores = ["#0082C1","#BFD3EB"]
    desfase = (0, 0)
    plt.pie(success_Rate, labels=nombres, autopct="%0.1f %%", colors=colores, explode=desfase)
    plt.axis("equal")
    
    plt.subplot(312)

    success_Rate = [successVSEM_nrsme, 100-successVSEM_nrsme]
    nombres = ["% of better image reconstruction     \ndeepPET Vs. EM                ",""]
    colores = ["#0082C1","#BFD3EB"]
    desfase = (0, 0)
    plt.pie(success_Rate, labels=nombres, autopct="%0.1f %%", colors=colores, explode=desfase)
    plt.axis("equal")
    
    plt.subplot(313)

    success_Rate = [successVSEMR_nrsme, 100-successVSEMR_nrsme]
    nombres = ["% of better image reconstruction     \ndeepPET Vs. EMR                ",""]
    colores = ["#0082C1","#BFD3EB"]
    desfase = (0, 0)
    plt.pie(success_Rate, labels=nombres, autopct="%0.1f %%", colors=colores, explode=desfase)
    plt.axis("equal")
             
    plt.show()
    
    # REPORT RESULTS
    
    option = str(input(' Do you want a report?'))
                 
    if option == 'yes':
        name_report = str(input('Name of report:'))
        model_used = str(input('The model tested was:'))
        set_used = str(input(' The set predicted was:'))
        
    
        file = open("/home/nrufo/Final_final/Reports/"+name_report, "w")
        file.write('The model tested was: '+model_used + os.linesep)
        file.write('The set predicted was: '+set_used + os.linesep)
        file.write(partdeep+partfbp+partEM+partEMR)
        file.close()
        print('Reported created! You can fin it in /home/nrufo/Final_final/Reports/')
    