'''Calculus Operation'''

import numpy


'''Differential of 2D array'''
#forward :  True means forward different
#           False means backward different
#bc : boundaly condition (default is neuman)
def diff2D(A,axis=0,forward=True,bc=None):
    A = numpy.array(A)
    if axis==1:
        A = numpy.transpose(A)
    if bc==None:
        if forward:
            B = numpy.append(A,A[:,-1],0)
            dx = B[:,1:] - A
        else:
            B = numpy.append(A[:,0],A,0)
            dx = A - B[:,:-1]
    else:
        bc = numpy.array(bc)
        if forward:
            B = numpy.append(A,bc,0)
            dx = B[:,1:] - A
        else:
            B = numpy.append(bc,A,0)
            dx = A - B[:,:-1]
    if axis==1:
        dx = numpy.transpose(dx)
    return dx
    
####################################################
    

