'''
Operation on grayscale image
'''
import numpy

'''adding noise'''
#img : image (array)
#percent : percent of noise (0-100)
#return : noisy image
def addnoise(img,percent=10):
    noise = numpy.random.normal(0,percent*0.85,img.shape)
    return numpy.uint8(numpy.clip(noise+numpy.float64(img),0,255))

##################################################

'''Denoising image by diffusion'''
#Find linear system Ax=b
#return A
def denoise_diff(m,n,alpha):
    
    A = numpy.zeros(((m*n,m*n)))

    for i in range(m):
        for j in range(n):
            C = [[ 0, -1,  0],
                 [-1,  4, -1],
                 [ 0, -1,  0]]
            if i==0 or i==m-1:
                C[1][1] -= 1
            if j==0 or j==n-1:
                C[1][1] -= 1
            T = numpy.zeros((m+2,n+2))
            T[i:i+3,j:j+3] = C
            Aij = T[1:-1,1:-1]
            A[i*n+j] = Aij.reshape(Aij.size)
    A = alpha*A+numpy.eye(m*n,m*n)
    return A

