'''
Operation on grayscale image
'''
import numpy
from scipy import sparse
from math import ceil, floor

'''adding noise'''
#img : image (array)
#percent : percent of noise (0-100)
#return : noisy image
def addnoise(img,std=1,per=0.05):
    noise = numpy.random.normal(0,std,img.shape)
    #z = u+k0*u*noise
    img_float = numpy.float64(img)
    noise_img = numpy.uint8(numpy.clip(img_float+per*img_float*noise,0,255))
    return noise_img

####################################################

#Find A for denoising diffusion
def diffusion_array(img_shape,alpha,tosparse=False):
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
#Affine transform with Ab = A*(img_coordinate)+b
def AFT(img,Ab):
    img = numpy.array(img)
    Ab = numpy.array(Ab)
    shape = img.shape
    height = shape[0]
    width = shape[1]
    TransImg = numpy.zeros(img.shape, numpy.uint8)
    for i in range(height):
        for j in range(width):
            coList = Ab[:,:-1].dot(numpy.array([i,j]))+Ab[:,-1]
            x,y = coList[0],coList[1]

            #Biliner Interpolation
            if x>=0 and x<=height-1 and y>=0 and y<=width-1:            
                cx = ceil(x)
                fx = floor(x)
                cy = ceil(y)
                fy = floor(y)
                if cx==fx:
                    dxf = 1/2
                    dxb = 1/2
                else:
                    dxf = cx-x
                    dxb = x-fx
                if cy==fy:
                    dyf = 1/2
                    dyb = 1/2
                else:
                    dyf = cy-y
                    dyb = y-fy
                TransImg[i][j] = dyb*dxb*img[cx][cy]+dyb*dxf*img[fx][cy]+dyf*dxb*img[cx][fy]+dyf*dxf*img[fx][fy]
    return TransImg    
    
####################################################

def warp_vec(img,vec_field,up=0,left=0,right=0,down=0):
    #shape of vec_field is (2,m+up+down,n+left+right) where m,n = imag.shape
    #extended up,left,right,down border
    shape = img.shape
    m,n = shape
    hight = m+up+down
    width = n+left+right
    WarpImg = numpy.zeros((hight,width))
    for i in range(hight):
        for j in range(width):
            x = i-up+vec_field[0,i,j]
            y = j-left+vec_field[1,i,j]

            #Biliner Interpolation
            if x>=0 and x<=m-1 and y>=0 and y<=n-1:            
                cx = ceil(x)
                fx = floor(x)
                cy = ceil(y)
                fy = floor(y)
                if cx==fx:
                    dxf = 1/2
                    dxb = 1/2
                else:
                    dxf = cx-x
                    dxb = x-fx
                if cy==fy:
                    dyf = 1/2
                    dyb = 1/2
                else:
                    dyf = cy-y
                    dyb = y-fy
                WarpImg[i][j] = dyb*dxb*img[cx][cy]+dyb*dxf*img[fx][cy]+dyf*dxb*img[cx][fy]+dyf*dxf*img[fx][fy]
    return WarpImg    
    
    
    
    
    
    
    
    