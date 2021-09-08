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

##################################################

#weighted Jacobi iteration method
def wJacobi(A,b,x0=None,Imax=10000,eps=1e-3,weight=1):
    t1 = time.time()
    if x0 is None:
        x0 = numpy.zeros_like(b)
    x = numpy.array(x0 ,dtype=numpy.float64)
    A = numpy.array(A)
    b = numpy.array(b)
    D = numpy.diag(numpy.diag(A))
    LU = D-A
    
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
        x = (LU.dot(x)+b)/(numpy.diag(D).reshape(b.shape)) #x=inv(D)*(LU*x+b)
        x = (1-weight)*x_old+weight*x
        diffSol = numpy.linalg.norm(x-x_old)
        residue = numpy.linalg.norm(b-A.dot(x))
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
def SOR(A,b,x0=None,Imax=10000,eps=1e-3,weight=1):
    t1 = time.time()
    if x0 is None:
        x0 = numpy.zeros_like(b)
    x = numpy.array(x0 ,dtype=numpy.float64)
    A = numpy.array(A)
    b = numpy.array(b)
    D = numpy.diag(numpy.diag(A))
    L = -numpy.tril(A,-1)
    U = -numpy.triu(A,1)
    invDL = numpy.linalg.inv(D-L)
    
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
        x = invDL.dot(U.dot(x)+b) #x=inv(D-L)*(Ux+b)        
        x = (1-weight)*x_old+weight*x
        diffSol = numpy.linalg.norm(x-x_old)
        residue = numpy.linalg.norm(b-A.dot(x))
        if diffSol<eps:
            print('solution dosen\'t change')
            return x
        if residue<eps:
            print('solution converges')
            return x
    print('exceed iteration')
    return x

##################################################

#Gradiant method
def GM(A,b,x0=None,Imax=10000,eps=1e-3):
    t1 = time.time()
    if x0 is None:
        x0 = numpy.zeros_like(b)
    x = numpy.array(x0 ,dtype=numpy.float64)
    A = numpy.array(A)
    b = numpy.array(b)
    h = b-A.dot(x)
    alpha = 1/max(abs(numpy.linalg.eigvals(A)))
    
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
        residue = numpy.linalg.norm(h)
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
        
        Ah = A.dot(h)
        hAh = numpy.dot(h,Ah)
        alpha = gg/hAh
        x = x + alpha*h
        g = g - alpha*Ah
        gg_old = gg.copy()
        gg = numpy.dot(g,g)
        h = g+h*gg/gg_old
        residue = gg
        if residue<eps:
            print('solution converges')
            return x
    print('exceed iteration')
    return x
    
##################################################