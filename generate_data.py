import glob
import cv2
import numpy as np
import nibabel as nib
import matplotlib
gan_path=glob.glob('/home/xin/PycharmProjects/GAN_repo/brats18_dataset/train/**/*_t2.nii.gz',recursive=True)
print(len(file_path))

t2=[]
tumor=0
nontumor=0
total=0
########################################################################################
for i in gan_path:

    t2_path=i
    label_path=i.replace('_t2','_seg')


    img_t2=nib.load(t2_path).get_data()
    label=nib.load(label_path).get_data()
    
    label[label!=0]=1
    
    img_t2_=img_t2.copy()
    img_t2_[img_t2_!=0]=1
    
    for j in range(154,-1,-1):
        a=label[:,:,j].sum()
        e=img_t2_[:,:,j].sum()
        threshold = 2000
        if  e>threshold:
            
            img_slice=img_t2[:,:,j]
            mm0=cv2.resize(img_slice,(128,128),interpolation=cv2.INTER_AREA)
            mm0=mm0/mm0.max()
            t2.append(mm0)

            total+=1
            if a>50:
                tumor+=1
            elif a==0:
                nontumor+=1                
# l_tumor.extend(l_nontumor)

np.save('/home/xin/PycharmProjects/GAN_repo/brats18_dataset/npy_gan/gan_t2.npy',t2)   



