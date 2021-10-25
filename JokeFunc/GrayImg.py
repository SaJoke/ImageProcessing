'''
Operation on grayscale image
'''
import numpy, cv2
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
    
####################################################

def grid_array(img_shape,nv=10,nh=10):
    m,n = img_shape
    grid = numpy.zeros((m,n))
    for i in range(1,nv):
        grid[i*m//nv,:] = numpy.ones_like(grid[i*m//nv,:])
    for j in range(1,nh):  
        grid[:,j*n//nh] = numpy.ones_like(grid[:,j*n//nh]) 
    return grid
    
def vec_field_array(img_shape,vec_field,nv=10,nh=10):
    m,n = img_shape
    vecField = 255*numpy.ones((m,n))
    
    for i in range(1,nv):
        for j in range(1,nh):
            tail = (floor(0.5+j*n//nh+vec_field[1,i*m//nv,j*n//nh]),floor(0.5+i*m//nv+vec_field[0,i*m//nv,j*n//nh]))
            head = (j*n//nh,i*m//nv)
            if not tail==head:
                cv2.arrowedLine(vecField,tail,head,(0,0,0),1,8,0,0.4)
    return vecField
    
    
    