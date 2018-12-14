# ContPy
ContPy is a simple Continuation library for python which allows complex arithmetic opetations.
The ideia behind ContPy is to provide a set a function to solving nonlinear continuation techniques which have the following format:

<a href="https://www.codecogs.com/eqnedit.php?latex=R(x,\alpha)=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?R(x,\alpha)=&space;0" title="R(x,\alpha)= 0" /></a>

given <a href="https://www.codecogs.com/eqnedit.php?latex=\alpha&space;=&space;[\alpha_0,\alpha_{end}]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha&space;=&space;[\alpha_0,\alpha_{end}]" title="\alpha = [\alpha_0,\alpha_{end}]" /></a>

Currently 3 modules are available in ContPy.
- optimize
- frequency
- operators
- utils

# Setup
In order to setup the contpy you should use git to clone the repo and the execute the setup.py

``` cmd
git clone https://github.com/jenovencio/contpy.git
cd COMPY
python setup develop
```



# Optimize module
The Optimize module is based on SciPy Optimize module, but wrapper were implemented in order 
to apply optimization techniques using complex functions, $f(x) \mathbb{C}$.
It uses the sample interface function as provided by SciPy:
- minimize
- root
- fsolve

But also provide a method for continuation technique:

``` python
from contpy import optimize 

a1, b1 = 0, 0.5
w = 2
xs = lambda s : (a1 + b1*s)*np.cos(w*s)
ys = lambda s : (a1 + b1*s)*np.sin(w*s)               
spiral = lambda s : (xs(s),ys(s)) 
spiral_res_vec = lambda x, s : np.array([(x[0] - (a1 + b1*s)*np.cos(w*s)), (x[1] - (a1 + b1*s)*np.sin(w*s))]) 

x0=np.array([0.0,0.0]) # initial guess
x_sol, p_sol, info_dict = optimize.continuation(spiral_res_vec,x0=x0,p_range=(-10.0,10.0),p0=0.0,max_dp=0.1,step=0.1,max_int=500)

```

# Frequency 
The frequency module provides some function to transform time to frequency domain and its inversion operations.
One of the main function is the `assemble_HBMOperator` which is a Truncated Fourier series. 
With this operator and the `create_Z_matrix` function Harmonic Balanced Method (HBM) can be easily performed.   

``` python
import matplotlib.pyplot as plt
import numpy as np
from contpy import frequency

#Target function 
coef = np.array([[10,0],
                 [0,3],
                 [0,1]])

freq_list = [1.0,3.0,5.0]

def func(t):
    fval = 0
    for i,f in enumerate(freq_list):
        fval += np.sum(coef[i].dot([np.cos(2.0*np.pi*f*t),np.sin(2.0*np.pi*f*t)]))
    return fval 
    
ndofs = 2
nH = 6 # number of harmonics to be considererd
n_points = 10000 # number of points to be considered

time_list = np.linspace(0,1,n_points)
f_list = np.array(list(map(func,time_list)))
f_desired = np.array([f_list]*ndofs)

Q = frequency.assemble_hbm_operator(ndofs,number_of_harm=nH ,n_points=n_points) # bases of truncaded Fourier

f_actual = Q.dot(Q.H.dot(f_desired))

plt.plot(f_list,'b',label='target')
plt.plot(f_actual[0],'r--',label='computed')
plt.legend()
plt.show()    

```


# AFT 
Altering Frequency and time domain (AFT) technique is easily implemented in ContPy. The HBMOpetator is a orthogonal operator, `Q`, which implies
its conjugate transpose `Q.H` is the inverse Fourier transform. Given a nonlinear function in time, `fnl(u)` the frequency nonlinear forced `fn_(u_)` is easy achieved by `Q` and `Q.H` multiplication.
`Q` is a special ContPy operator and developer can add more functioned in this class.

``` python
fnl = lambda u : beta*(Tc.dot(u)**3)
fnl_ = lambda u_ : Q.H.dot(fnl(Q.dot(u_))) - fl_
```