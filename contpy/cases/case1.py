import numpy as np

name =  '2 dof Duffing oscillator.'

#HBM variables
n_dofs = 2

# case variables
amplitude_dof_1 = 5.0
amplitude_dof_2 = 0.0
P = np.array([amplitude_dof_1, amplitude_dof_2], dtype = np.complex)
beta = 5.0
m1 = 1.0
m2 = m1
k1 = 1.0
k2 = k1
c1 = 0.05
c2 = c1

K = np.array([[k1+k2, -k2],
              [-k2,k2+k1]])

M = np.array([[m1,0.0],
              [0.0,m2]])

C = np.array([[c1+c2,-c2],
              [-c2,c1+c2]])

B_delta = np.array([[-1, 1],
                    [-1, 1]])

H = np.array([[-1, 0],
              [ 0, 1]])

Tc = H.dot(B_delta)




