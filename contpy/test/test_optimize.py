import sys 
from scipy import optimize, sparse
import numpy as np
from unittest import TestCase, main
from scipy.misc import derivative
from scipy.sparse.linalg import LinearOperator
from scipy import linalg
import numdifftools as nd
import matplotlib.pyplot as plt

sys.path.append('..')
from src.frequency import assemble_jacobian_Zw_matrix, assemble_HBMOperator, create_Z_matrix
from src.operators import ReshapeOperator
from src import optimize

class  Test_optimize(TestCase):
    def setUp(self):
        pass

    def test_duffing_with_analytical_jac(self):
        from cases.case2 import name, n_dofs, K,M,C,P,beta, B_delta, H, Tc

        #HBM variables
        nH = 1
        n_points = 100

        # buiding Harmonic bases and the Augmented force vector amplitude
        Q = assemble_HBMOperator(n_dofs,number_of_harm=nH ,n_points=n_points)
        P_aug = list(0*P)*nH
        P_aug[0:n_dofs] = list(P)
        P_aug = np.array(P_aug)
        fl = Q.dot(P_aug).real

        # building Residual equation for continuation
        beta = 1.0
        fl_ = Q.H.dot(fl) # force in frequency domain
        fnl = lambda u : beta*(Tc.dot(u)**3)
        fnl_ = lambda u_ : Q.H.dot(fnl(Q.dot(u_))) - fl_
        Z = lambda w :create_Z_matrix(K,C,M,f0= w/(2.0*np.pi),nH=nH)
        R = lambda u_, w : Z(w).dot(u_) + fnl_(u_)

        #comptuting analytical derivatives
        Ro = ReshapeOperator(n_dofs,n_points)
        JZw = lambda w : assemble_jacobian_Zw_matrix(K,C,M,f0= w/(2.0*np.pi),nH=nH)
        Jfnl = lambda u : np.diag(Ro.T.dot(3*beta*(Tc.dot(u)**2)))
        Jfln_ = lambda u_ :  Q.H.Q@Jfnl(Q.dot(u_))@Q.Q
        JRu_ = lambda w : lambda u_ :  Z(w) + Jfln_(u_)
        JRw = lambda u_ : lambda w : np.array([JZw(w).dot(u_)])

        # computing numerical jacobian
        JRw_num = lambda u_ : optimize.real_jacobian(lambda w : R(u_,w))
        JRu_num = lambda w : optimize.complex_jacobian(lambda u_ : R(u_,w))

        # solving continuation with analytical derivatives
        x0 = np.array([0.0]*n_dofs*nH,dtype=np.complex)
        p0 = 0.1
        y_d, p_d, info_dict = optimize.continuation(R,x0=x0,p_range=(0.01,3.0), p0=p0, correction_method='matcont',
                                                    jacx=JRu_,#jacp=JRw_num,
                                                    max_int=2000, max_dp=0.05,step=0.1, max_int_corr=20, tol=1.0E-10)


    def test_analytic_jac(self):
        from cases.case2 import name, n_dofs, K,M,C,P,beta, B_delta, H, Tc

        #HBM variables
        nH = 1
        n_points = 100

        # buiding Harmonic bases and the Augmented force vector amplitude
        Q = assemble_HBMOperator(n_dofs,number_of_harm=nH ,n_points=n_points)
        P_aug = list(0*P)*nH
        P_aug[0:n_dofs] = list(P)
        P_aug = np.array(P_aug)
        fl = Q.dot(P_aug).real

        # building Residual equation for continuation
        beta = 1.0
        fl_ = Q.H.dot(fl) # force in frequency domain
        fnl = lambda u : beta*(Tc.dot(u)**3)
        fnl_ = lambda u_ : Q.H.dot(fnl(Q.dot(u_))) - fl_
        Z = lambda w :create_Z_matrix(K,C,M,f0= w/(2.0*np.pi),nH=nH)
        R = lambda u_, w : Z(w).dot(u_) + fnl_(u_)

        #comptuting analytical derivatives
        Ro = ReshapeOperator(n_dofs,n_points)
        JZw = lambda w : assemble_jacobian_Zw_matrix(K,C,M,f0= w/(2.0*np.pi),nH=nH)
        Jfnl = lambda u : np.diag(Ro.T.dot(3*beta*(Tc.dot(u)**2)))
        Jfln_ = lambda u_ :  Q.Q.T@Jfnl(Q.dot(u_))@Q.Q.conj()
        JRu_ = lambda w : lambda u_ :  Z(w) + Jfln_(u_)
        JRw = lambda u_ : lambda w : np.array([JZw(w).dot(u_)])

        # computing numerical jacobian
        JRw_num = lambda u_ : optimize.real_jacobian(lambda w : R(u_,w))
        JRu_num = lambda w : optimize.complex_jacobian(lambda u_ : R(u_,w))

        # solving continuation with analytical derivatives
        x0 = np.array([0.0]*n_dofs*nH,dtype=np.complex)
        p0 = 0.1

        fun = R
        jacx = JRu_
        fun_real = lambda x, p : optimize.func_wrapper(fun, x, p)
        x0_real = optimize.complex_array_to_real(x0)
            
        # creating parametric functions
        Fx = lambda p : lambda x : fun_real(x,p)
        Fp = lambda x : lambda p : fun_real(x,p)

        JFx_num = lambda p : nd.Jacobian(Fx(p))
        JFx = lambda p : lambda x : optimize.jac_wrapper(jacx,x,p).toarray() 

        n_test = 3
        for i in range(n_test):
            xn = np.random.rand(n_dofs*nH*2)
            xn.dtype=np.complex
            xn_real = optimize.complex_array_to_real(xn)
            pn = 10*np.random.rand()
            J_num_eval =  JFx_num(pn)(xn_real)
            J_ana_eval = JFx(pn)(x0_real)

            J_num_eval_complex = optimize.real_block_matrix_to_complex(J_num_eval)
            J_ana_eval_complex = optimize.real_block_matrix_to_complex(J_ana_eval)

            J_num_real = J_num_eval_complex.real
            J_num_imag = J_num_eval_complex.imag

            J_ana_real = J_ana_eval_complex.real
            J_ana_imag = J_ana_eval_complex.imag

            norm_real = np.linalg.norm(J_ana_real.flatten())
            norm_imag = np.linalg.norm(J_ana_imag.flatten())

            np.testing.assert_array_almost_equal(J_ana_real.flatten()/norm_real,  J_num_real.flatten()/norm_real ,  decimal=10 )
            np.testing.assert_array_almost_equal(np.abs(J_ana_imag).flatten()/norm_imag ,  np.abs(J_num_imag).flatten()/norm_imag  ,  decimal=10 )


        JFp_num = lambda x : nd.Jacobian(Fp(x))
        JFp =  lambda x : lambda p : optimize.jac_wrapper(jacp,np.array([p]),x).toarray() 
          

if __name__ == '__main__':
    #main()
    
    test_obj = Test_optimize()
    #test_obj.test_duffing_with_analytical_jac()
    test_obj.test_analytic_jac()
