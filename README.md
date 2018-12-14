# ContPy
ContPy is a simple Continuation library for python which allows complex arithmetic opetations.
The ideia behind ContPy is to provide a set a function to solving nonlinear continuation techniques which have the following format:

<a href="https://www.codecogs.com/eqnedit.php?latex=R(x,\alpha)=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?R(x,\alpha)=&space;0" title="R(x,\alpha)= 0" /></a>

given \alpha = [\alpha_0,\alpha_{end}]

Currently 3 modules are available in ContPy.
- optimize
- frequency
- operators

#Optimize module
The Optimize module is based on SciPy Optimize module, but wrapper were implemented in order 
to apply optimization techniques using complex functions, $f(x) \mathbb{C}$.
It uses the sample interface function as provided by Scipy:
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


