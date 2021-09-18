'''
Operation on grayscale image
'''
import numpy
from scipy import sparse

'''adding noise'''
#img : image (array)
#percent : percent of noise (0-100)
#return : noisy image
def addnoise(img,percent=10):
    noise = numpy.random.normal(0,percent*0.85,img.shape)
    return numpy.uint8(numpy.clip(noise+numpy.float64(img),0,255))

####################################################

#Find A for denoising diffusion
def denoise_diff(img_shape,alpha,tosparse=False):
    m , n = img_shape
    Ac = 4*alpha*numpy.ones((m,n))+1
    Au = -alpha*numpy.ones((m,n))
    Al = -alpha*numpy.ones((m,n))
    Ac[0,:] -= alpha
    Ac[:,0] -= alpha
    Ac[:,-1] -= alpha
    Ac[-1,:] -= alpha
    Al[:,0] = 0
    Ac = Ac.reshape(Ac.size)
    Al = Al.reshape(Al.size)[1:]
    Au = Au.reshape(Au.size)[n:]
    Ar = Al[::-1]
    Ad = Au[::-1]
    A = sparse.diags([Ac,Au,Al,Ar,Ad],[0,-n,-1,1,n])
    if not tosparse:
        return A.toarray()
    return A
    
####################################################

#def PSNR(img1,img2):
    
    
    
    
    
    
    
    
    
    
    
    