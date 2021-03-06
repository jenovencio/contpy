import numpy as np
import scipy.sparse as sparse
import collections
from unittest import TestCase, main
from numpy.testing import assert_array_equal
from scipy.sparse.linalg import LinearOperator as LO
from scipy import sparse
from .utils import OrderedSet, get_dofs

class ReshapeOperator():
    ''' Reshape operator acts as a LinearOperator
    reshaping an array based on kroneker product
    '''
    def __init__(self,ndofs,n_points):
        self.ndofs = ndofs
        self.n_points = n_points

    def dot(self,u):
        return self._matvec(u)
    
    def _matvec(self,u):
        if len(u) == self.ndofs*self.n_points:
            return u.reshape(self.ndofs,self.n_points)
        else:
            raise ValueError('Not compatible array')
            
    def _transpose(self):
        return ReshapeOperatorTranspose(self.n_points,self.ndofs,self)
    
    @property
    def shape(self):
        return (self.ndofs,self.n_points,self.ndofs*self.n_points)
    
    @property
    def T(self):
        return self._transpose()
        
class ReshapeOperatorTranspose():
    def __init__(self, n_points,ndofs,res_opt_obj=None):
        self.ndofs = ndofs
        self.n_points = n_points
        self.res_opt_obj = res_opt_obj

    def _matvec(self,u):
        if u.shape == (self.ndofs, self.n_points):
            return u.flatten()
        else:
            raise ValueError('Not compatible array')
    
    def _transpose(self):
        if self.res_opt_obj is None:
            return ReshapeOperator(self.n_points,self.ndofs)
        else:
            return self.res_opt_obj
    
    def dot(self,u):
        return self._matvec(u)
    
    @property
    def shape(self):
        return (self.ndofs*self.n_points,self.n_points,self.ndofs)
    
    @property
    def T(self):
        return self._transpose()
            
class HBMOperator():
    def __init__(self,ndofs,q):
        self.ndofs = ndofs
        self.npoints, self.nH = q.shape
        self.dtype = q.dtype
        self.shape = (ndofs,self.npoints,self.nH*self.ndofs)
        self._build_Q(ndofs,q)
        self.Ro = ReshapeOperator(self.ndofs,self.npoints)
        self.HBM_obj = HBMOperator_(self.Q,self.Ro)
        self.adjoint = HBMOperatorAdjoint_(self.Q.conj(),self.Ro, self)
        
    def _build_Q(self,ndofs,q):
        I = sparse.eye(ndofs)
        Q = sparse.kron(I,q[:,0])
        for i in range(1,self.nH):
            Q = sparse.vstack([Q,sparse.kron(I,q[:,i])])
        self.Q = Q.T
        
    def _matvec(self,u_):
        return self.HBM_obj._matvec(u_)
        
    def _transpose(self):
        return HBMOperatorTranspose_(self.Q,self.Ro, self)
    
    def conj(self):
        return HBMOperator_(self.Q.conj(),self.Ro)
    
    def _adjoint(self):
        return  self.adjoint

    @property
    def T(self):
        return self._transpose()
        
    @property
    def H(self):
        return self._adjoint()
    
    def dot(self,u_):
        return self._matvec(u_)

class HBMOperator_():
    def __init__(self,Q,Ro):
        
        self.Ro = Ro
        self.Q = Q
        self.ndofs, self.npoints, total_dof = Ro.shape
        total_dof, nH_x_ndofs = Q.shape
        self.nH = int(nH_x_ndofs/self.ndofs)
        self.transpose = None
    
    @property
    def shape(self):
        return (self.ndofs,self.npoints,self.nH*self.ndofs)

    def _matvec(self,u_):
        return self.Ro.dot(self.Q.dot(u_).real)
    
    def _transpose(self):
        return self.transpose

    def conj(self):
        None

    def _adjoint(self):
        return self._transpose.conj()

    @property
    def T(self):
        T = self._transpose()
        if T is None:
            return self
        else:
            return T
    
    @property
    def H(self):
        return self._adjoint
    
    def dot(self,u_):
        return self._matvec(u_)

class HBMOperatorTranspose_(HBMOperator_):
    def __init__(self,Q,Ro,transpose = None):
        super(HBMOperatorTranspose_,self).__init__(Q,Ro)
        
        self.Q = self.Q.T
        self.Ro = self.Ro.T
        self.transpose = transpose
        self._conj = None

    @property
    def shape(self):
        return (self.nH*self.ndofs,self.npoints,self.ndofs)

    def _matvec(self,u):
        return self.Q.dot(self.Ro.dot(u))

    def _transpose(self):
        return self.transpose
    
    def conj(self):
       return None

class HBMOperatorAdjoint_(HBMOperator_):
    def __init__(self,Q,Ro, adjoint = None):
        super(HBMOperatorAdjoint_,self).__init__(Q,Ro)
        
        self.Q = self.Q.T
        self.Ro = self.Ro.T
        self.adjoint = adjoint
        self._conj = None

    @property
    def shape(self):
        return (self.nH*self.ndofs,self.npoints,self.ndofs)

    def _matvec(self,u):
        return self.Q.dot(self.Ro.dot(u))

    def _transpose(self):
        return None
    
    def conj(self):
       return None

    def _adjoint(self):
       return self.adjoint

class HBMOperator_conj_():
    def __init__(self,Q,Ro):
        super(HBMOperator_,self).__init__(Q,Ro)
        self.Q = self.Q.conj()
        
    def _matvec(self,u):
        return self.Q.T.dot(self.Ro.T.dot(u))

class Nonlinear_Force_AFT():
    def __init__(self,Q,Nonlinear_Force_obj):
        self.Q = Q
        self.Nonlinear_Force_obj = Nonlinear_Force_obj
        self.f_nonlinear = Nonlinear_Force_obj.compute_force
        self.J_nonlinear = Nonlinear_Force_obj.compute_jacobian
        self.Fnl_obj_list = Nonlinear_Force_obj.Fnl_obj_list
        self.row_indices = np.array([],dtype=np.int)
        self.col_indices = np.array([],dtype=np.int)
        self.jacobian_indices_computed = False
        self.ndofs, self.time_points, ndofs_x_nH = Q.shape
        self.nH = int(ndofs_x_nH/self.ndofs)
        self.Q_block_list = self._create_Q_block_list()

    def _create_Q_block_list(self):
        ndofs = self.ndofs
        nH = self.nH
        QQA_ = self.Q.Q.tocsr()
        Q_block_list = []
        for i in range(nH):
            QQA = QQA_[:,i*ndofs:(i+1)*ndofs]
            Q_block_list.append(QQA)
        return Q_block_list

    def _multiply(self,JJJ_s,Q_block_list):
        nH = self.nH
        J_list = []
        J_conj_list = []
        for i in range(nH):
            row_list = []
            conj_row_list = []
            for j in range(nH):
            
                QQA = Q_block_list[i]
                QQT = Q_block_list[j].T
                QQH = QQT.conj()
                JQA = JJJ_s.dot(QQA)
                J_row = QQH.dot(JQA)
                J_conj_row = QQT.dot(JQA)
                row_list.append(J_row)
                conj_row_list.append(J_conj_row)
                
            J_list.append(row_list)
            J_conj_list.append(conj_row_list)

        return J_list, J_conj_list

    def compute_force_and_jac(self,u_,compute_jac=True):
        ''' evaluate the nonlinear force in frequency domain
        '''
        time_points = self.time_points 
        ndofs = self.ndofs
        Q = self.Q

        if compute_jac:
            fnl_in_time, J_in_time = self._fnl_and_jac(Q.dot(u_),compute_jac=compute_jac)
        else:
            fnl_in_time = self._fnl_and_jac(Q.dot(u_),compute_jac=compute_jac)
            
        fnl_in_freq = Q.H.dot(fnl_in_time)

        if compute_jac:
            self.JJJ = JJJ = J_in_time.reshape(time_points*ndofs,time_points*ndofs,order='C')
    
            J_list, J_conj_list = self._multiply(JJJ,self.Q_block_list)

            jac_in_freq_complex = sparse.bmat(J_list)
            jac_in_freq_complex_conjugate = sparse.bmat(J_conj_list)
                
            return fnl_in_freq, jac_in_freq_complex, jac_in_freq_complex_conjugate
        
        else:
            return fnl_in_freq


    def compute_force(self,u_):
        return self.compute_force_and_jac(u_,compute_jac=False)

    def local2global(self,local_array,m,i): 
        return local_array*m + i

    def _fnl_and_jac(self,U,compute_jac=True):
        
        Fnl_obj_list = self.Fnl_obj_list
        n, m = self.ndofs, self.time_points
        Fnl, JFnl = self.f_nonlinear, self.J_nonlinear
        big_n = m*n
        global_force = np.zeros((n,m))
        row_indices = self.row_indices
        col_indices = self.col_indices
        local2global = self.local2global
        data = np.array([])
        for i in range(self.time_points):
            un_1 = U[:,i]
            if i>1:
                un = U[:,i-1]
            else:
                un = 0.0*un_1

            fnl_eval = Fnl(un_1,un)
            
            global_force[:,i] = fnl_eval
           
            if compute_jac:
                jfnl_eval = JFnl(un_1,un)
                jfnl_eval_copy = jfnl_eval.tocoo()
                if not self.jacobian_indices_computed:
                    
                    rows_local = jfnl_eval_copy.row
                    cols_local = jfnl_eval_copy.col
                    rows = local2global(rows_local,m,i)
                    cols = local2global(cols_local,m,i)
                    row_indices =  np.concatenate((row_indices,rows))
                    col_indices =  np.concatenate((col_indices,cols))

                data = np.concatenate((data,jfnl_eval_copy.data))

            for Fnl_obj in Fnl_obj_list:
                Fnl_obj.refresh_alpha()

        for Fnl_obj in Fnl_obj_list:
                Fnl_obj.refresh_alpha()
        
        if compute_jac:
            self._J_aloc = sparse.coo_matrix((data,(row_indices,col_indices)),shape=(big_n,big_n))
            return global_force, self._J_aloc.tocsc()
        else:
            return global_force
    
class SelectionOperator():
    def __init__(self,selection_dict,id_matrix=None):
        ''' the selection dict contain labels as key and 
        dofs as values. The idea is to provide a class
        which can apply permutation in matrix and also global to local map
        
        parameters 
            selection_dict : Ordered dict
                dict with string and dofs
        
        '''
        self.selection_dict = selection_dict
        self.all_keys_set = OrderedSet(self.selection_dict.keys())
        self.removed_keys = OrderedSet()
        self.red_dof_dict = None
        self.local_indexes = []
        self.bounds = {}
        self.length = {}
        self.local_to_global_dof_dict = {} 
        self.global_to_local_dof_dict = {} 
        self.global_id_matrix = id_matrix
        self.removed_dict_values = {}

        count = 0
        local_dof_counter = 0
        for key, dof_list in selection_dict.items():
            self.local_indexes.extend(dof_list)
            length = len(self.local_indexes)
            self.length[key] = len(dof_list)
            self.bounds[key] = [count,length]
            count += length
            for value in dof_list:
                self.local_to_global_dof_dict[local_dof_counter] = value
                self.global_to_local_dof_dict[value] = local_dof_counter
                local_dof_counter += 1
        
        
        self.P = self.create_permutation_matrix(self.local_indexes)
        self.ndof = max(self.P.shape)
    
    def nodes_to_local_dofs(self):
        pass
    
    def create_permutation_matrix(self,local_indexes):
        ''' create a Permutation matrix based on local id
        
        '''
        ndof = len(local_indexes)
        P = sparse.lil_matrix((ndof, ndof), dtype=np.int8)
        P[local_indexes, np.arange(ndof)] = 1
        return P.T.tocsc()
        
    def create_block_matrix(self,M):
        ''' This function create block matrix with string
        which is useful for applying boundary conditions
        '''
        block_matrix = {}
        for key1, dofs_1 in self.selection_dict.items():
            for key2, dofs_2 in self.selection_dict.items():
                block_matrix[key1,key2] = M[np.ix_(dofs_1, dofs_2)]
        
        return block_matrix
        
    def create_block_vector(self,f):
        block_vector = {}
        for key1, dofs_1 in self.selection_dict.items():
            block_vector[key1] = f[dofs_1]
        
        return block_vector
    
    def assemble_matrix(self,M,list_of_strings):
        ''' This method assemble a matrix based on the list of string
        useful for ordering the matrix according to the block string matrix
        paramenter:
            M : np.array
                matrix to be reordered
            list of strings : list
                list with a sequence of string which gives the 
                order of the degrees of freedom associated with M11
            
            return a ordered Matrix
            
            ex. 
                M_block = s.create_block_matrix(M)
                M_row1 = sparse.hstack((M_block['l','l'],M_block['l','h'],M_block['l','i']))
                M_row2 = sparse.hstack((M_block['h','l'],M_block['h','h'],M_block['h','i']))
                M_row3 = sparse.hstack((M_block['i','l'],M_block['i','h'],M_block['i','i']))
                M_sector = sparse.vstack((M_row1,M_row2,M_row3)).tocsc()
            
            
        '''
        self.create_reduced_selector(list_of_strings)
        
        for key in self.removed_keys:
            print('Removing dofs related to label %s. Setting dof values equal to 0' %key)
            self.removed_dict_values[key] = np.zeros(len(self.selection_dict[key]))

        M_block = self.create_block_matrix(M)
        
        M_rows = []
        for s_i in list_of_strings:
            M_row_j_list = [] 
            for s_j in list_of_strings:
                M_row_j_list.append(M_block[s_i,s_j])
            M_rows.append(sparse.hstack(M_row_j_list))
        
        return sparse.vstack(M_rows).tocsc()
          
    def assemble_vector(self,f,list_of_strings):
        ''' This method assemble a vector based on the list of string
        useful for ordering the matrix according to the block string matrix
        paramenter:
            M : np.array
                1-d array to be reordered
            
            list of strings : list
                list with a sequence of string which gives the 
                order of the degrees of freedom associated with M11
            
            return a ordered Matrix
            
            
        '''
        
        f_block = self.create_block_vector(f)
        
        f_rows = np.array([])
        for s_i in list_of_strings:
            f_rows = np.append(f_rows, f_block[s_i])
        
        return f_rows
        
    def select_block(self,M,rows,columns):
        pass
    
    def create_reduced_selector(self,list_of_strings):
        
        self.removed_keys =  self.all_keys_set - list_of_strings # copy list with all keys
        self.red_dof_dict = collections.OrderedDict()
        init_dof = 0
        for key in list_of_strings:
            last_dof = init_dof + len(self.selection_dict[key])
            self.red_dof_dict[key] = np.arange(init_dof,last_dof) 
            init_dof = last_dof
        
        self.reduced_selector = SelectionOperator(self.red_dof_dict,self.global_id_matrix)
        
    def build_B(self,label):
        ''' Build Boolean selection operator
        
        '''
        
        
        #local_id = []
        #for node_id in node_list:
        #    global_dof_list = self.global_id_matrix[node_id]
        #    for global_dof in global_dof_list:
        #        local_dof = self.global_to_local_dof_dict[global_dof]
        #        local_id.append(local_dof)
        
        local_id = self.selection_dict[label]
        B = sparse.csc_matrix((len(local_id), self.ndof), dtype=np.int8)
        B[np.arange(len(local_id)), local_id ] = 1
        return B     
    
    def assemble_global_vector(self,u):
        ''' This function assembles the global vector 
        based on the solution of the reduced system based on 
        the assemble_block_matrix

        Parameters:
            u : np.array
                solution of the reduced system

        return :
            np.array
                solution of the global system
        '''

        u_global = np.array([])
        for key,values in self.removed_dict_values.items():
            u_global = np.concatenate((u_global, values))
            
        for key, values_id in self.reduced_selector.selection_dict.items():
            u_global = np.concatenate((u_global, u[values_id]))

        return self.P.T.dot(u_global) # back to original order
            
def build_SelectionOperator(comp,map_dict,tag_name='phys_group'):
    ''' build a SelectionOperator given a AMfe mechanical System
    
    Parameters:
        comp : Amfe.MechanicalSystem
            an Amfe.MechanicalSystem instance
        
        map_dict : dict
            a dict containing a mapping of new names 
            ex.: map_dict = {'d': {'tag_value': 1, 'direction' : 'xy'},
            'n': {'tag_value': 12, 'direction' : 'y'},
            'c': {'tag_value': 12, 'direction' : 'y'}}
        tag_name : string
            tag name to select dofs
    '''
    m = comp.mesh_class
    id_matrix = comp.assembly_class.id_matrix
    
    all_dofs = []
    for key, item in id_matrix.items():
        all_dofs.extend(item)
    
    dof_dict = collections.OrderedDict()
    all_label_dofs = []
    for key, item in map_dict.items():
        submesh_obj = m.get_submesh(tag_name, item['tag_value'])
        dofs_dict = OrderedSet(comp.get_dofs(submesh_obj, direction =item['direction']))
        dofs_list = list(dofs_dict - all_label_dofs) # eliminating intersection with already existing dofs
        
        #update with COPY priorit_dofs 
        dof_dict[key] = dofs_list
        all_label_dofs.extend(dofs_list)
        
    interior_dofs = list(OrderedSet(all_dofs) - all_label_dofs)
    dof_dict['i'] = interior_dofs

    return SelectionOperator(dof_dict,id_matrix)
            
            
class  Test_Operators(TestCase):
    def setUp(self):
        pass
    
    def test_ReshapeOperator(self):
        ndofs = 3
        n_points = 5

        u = np.array([1,2,3,4,5]*ndofs)
        u[5] = -8
        u[10] = -7
        u[14] = -9
        
        desired = np.array([[ 1,  2,  3,  4,  5],
                            [-8,  2,  3,  4,  5],
                            [-7,  2,  3,  4, -9]])
        
        R = ReshapeOperator(ndofs,n_points)
        actual = R.dot(u)
        u_back = R.T.dot(desired)
        
        assert_array_equal(actual,desired)
        assert_array_equal(u_back,u )
    
    def test_HBM_Operator(self):
        pass

if __name__ == '__main__':
    main()            