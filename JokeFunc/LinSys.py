"""
solving a linear system Ax = b

A : matrix (list or ndarray)
x : solution vector
b : vector  (list or ndarray)
x0 : initial solution for iteration method (list or ndarray)
Imax : maximum of iterations (integer)
eps : small number for stop process  (floating)
weight : used for weighted Jacobi and SOR (floating)
"""

import numpy,time
from scipy import sparse
from scipy.sparse import linalg

##################################################

#weighted Jacobi iteration method
def wJacobi(A,b,x0=None,Imax=10000,eps=1e-3,weight=1):
    t1 = time.time()
    if x0 is None:
        x0 = numpy.zeros_like(b)
    x = numpy.array(x0 ,dtype=numpy.float32)
    A = numpy.array(A)
    b = numpy.array(b)
    D = numpy.diag(numpy.diag(A))
    LU = D-A #L+U
    invD = numpy.diag(1/numpy.diag(D).reshape(b.shape))
    T = invD.dot(LU)
    c = invD.dot(b)
    res0 = numpy.linalg.norm(b-A.dot(x))
    
    # option : approximate exceeded time (part1)
    count = 0
    t2 = time.time()
    p = False
        
    for i in range(Imax):
        
        # option : approximate exceeded time (part2)
        if not p:
            if time.time()-t2>3:
                p = True
                et = Imax*(time.time()-t2)/count
                print('max time ~',et+t2-t1) #expected time until exceed iteration
        count+=1
        
        x_old = x.copy()
        x = T.dot(x)+c 
        x = (1-weight)*x_old+weight*x
        diffSol = numpy.linalg.norm(x-x_old)
        residue = numpy.linalg.norm(b-A.dot(x))/res0
        print(i+1,diffSol,'res=',residue)
        if diffSol<eps:
            print('solution dosen\'t change')
            return x
        if residue<eps:
            print('solution converges')
            return x
    print('exceed iteration')
    return x

##################################################

#weighted Gauss-Seidel iteration method(SOR)
def SOR(A,b,x0=None,Imax=1000,eps=1e-3,weight=1,progress=False):
    t1 = time.time()
    if x0 is None:
        x0 = numpy.zeros_like(b)
    x = numpy.array(x0 ,dtype=numpy.float16)
    b = numpy.array(b)
    if sparse.issparse(A):
        D = sparse.diags(A.diagonal())
        L = -sparse.tril(A,-1)
        U = -sparse.triu(A,1)
        invDL = linalg.spsolve_triangular(D-L,numpy.eye(len(b)))
        U = U.toarray()
    else:
        A = numpy.array(A)
        D = numpy.diag(numpy.diag(A))
        L = -numpy.tril(A,-1)
        U = -numpy.triu(A,1)
        invDL = numpy.linalg.inv(D-L)
    
    T = invDL.dot(U)
    c = invDL.dot(b)
    res0 = numpy.linalg.norm(b-A.dot(x))
    # option : approximate exceeded time (part1)
    count = 0
    p = False
    t2 = time.time()
    
    for k in range(Imax):

        # option : approximate exceeded time (part2)
        if not p:
            if time.time()-t2>3:
                p = True
                et = Imax*(time.time()-t2)/count
                print('max time ~',et+t2-t1) #expected time until exceed iteration
        count+=1
        
        x_old = x.copy()
        x = T.dot(x)+c #x=inv(D-L)*(Ux+b)        
        x = (1-weight)*x_old+weight*x
        diffSol = numpy.linalg.norm(x-x_old)
        residue = numpy.linalg.norm(b-A.dot(x))/res0
        #print(k+1,diffSol,'res=',residue)
        if progress:
            if (k+1)%(100*Imax//len(b)+1)==0:
                print(k+1,'diffSol = ',diffSol,'residue = ',residue)
        if diffSol<eps:
            print('SOR solution dosen\'t change')
            return x
        if residue<eps:
            print('SOR solution converges')
            return x
    print('SOR exceed iteration')
    return x

##################################################

#Gradiant method
def GM(A,b,x0=None,Imax=1000,eps=1e-3):
    t1 = time.time()
    if x0 is None:
        x0 = numpy.zeros_like(b)
    x = numpy.array(x0 ,dtype=numpy.float32)
    A = numpy.array(A)
    b = numpy.array(b)
    h = b-A.dot(x)
    alpha = 1/max(abs(numpy.linalg.eigvals(A)))
    res0 = numpy.linalg.norm(b-A.dot(x))
    # option : approximate exceeded time (part1)
    count = 0
    p = False
    t2 = time.time()
    
    for k in range(Imax):
        
        # option : approximate exceeded time (part2)
        if not p:
            if time.time()-t2>3:
                p = True
                et = Imax*(time.time()-t2)/count
                print('max time ~',et+t2-t1) #expected time until exceed iteration
        count+=1
        
        x_old = x.copy()
        x = x + alpha*h
        h = b-A.dot(x)
        diffSol = numpy.linalg.norm(x-x_old)
        residue = numpy.linalg.norm(h)/res0
        print(k+1,diffSol,'res=',residue)
        if diffSol<eps:
            print('solution dosen\'t change')
            return x
        if residue<eps:
            print('solution converges')
            return x
    print('exceed iteration')
    return x

##################################################

#conjugate gradiant method
def CG(A,b,x0=None,Imax=10000,eps=1e-3):
    t1 = time.time()
    if x0 is None:
        x0 = numpy.zeros_like(b)
    x = numpy.array(x0 ,dtype=numpy.float64)
    A = numpy.array(A)
    b = numpy.array(b)
    g = b - A.dot(x)
    h = g.copy()
    gg = numpy.dot(g,g)
    res0 = numpy.linalg.norm(b-A.dot(x))
    
    # option : approximate exceeded time (part1)
    count = 0
    p = False
    t2 = time.time()
    for k in range(Imax):
        # option : approximate exceeded time (part2)
        if not p:
            if time.time()-t2>1:
                p = True
                et = Imax*(time.time()-t2)/count
                print('max time ~',et+t2-t1) #expected time until exceed iteration
        count+=1
        
        x_old = x.copy()
        Ah = A.dot(h)
        hAh = numpy.dot(h,Ah)
        alpha = gg/hAh
        x = x + alpha*h
        g = g - alpha*Ah
        gg_old = gg.copy()
        gg = numpy.dot(g,g)
        h = g+h*gg/gg_old
        #residue = gg
        diffSol = numpy.linalg.norm(x-x_old)
        residue = numpy.linalg.norm(b-A.dot(x))/res0
        #print(k+1,diffSol,'CG_res=',residue)
        if diffSol<eps:
            print('solution dosen\'t change')
            return x
        if residue<eps:
            print('solution converges')
            return x
    print('exceed iteration')
    return x
    
##################################################