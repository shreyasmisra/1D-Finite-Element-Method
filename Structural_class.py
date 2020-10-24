from numpy import *

# Program is used for finding the unknowne stresses and strains in a system consisting of bar elements. It takes into consideration point and thermal forces.
# The boundary conditions are fixed or a displacement is specified. 
# Assumued element connectivity --- 1 -> 2 -> 3 ...... -> N.

class Structural1D:
    def __init__(self,E,A,l,n_elements,temperature_effect = False):
        self.E = E
        self.l = l
        self.A = A
        self.n_elements = n_elements
        self.n_nodes = n_elements+1
        self.L = self.l[0]*self.n_elements
        self.temperature_effect = temperature_effect

    def find_stiffness_matrix(self):
        '''
        Calculates the Global stiffness matrix based on Minimum Potential Energy Principle. 
        K is the global stiffness matrix of shape - number of nodes x number of nodes. Assuming 1 DoF at each node
        '''
        def stiffness(l,A,E):
                return (E*A)/l
            
        self.K = zeros((self.n_nodes,self.n_nodes))
        for i in range(self.n_elements):
            self.K[i,i+1] = -1*stiffness(self.l[i],self.A[i],self.E[i])
            self.K[i+1,i] = -1*stiffness(self.l[i],self.A[i],self.E[i])
            if i == 0:
                self.K[0,0] = stiffness(self.l[i],self.A[0],self.E[i])
                self.K[-1,-1] = stiffness(self.l[-1],self.A[-1],self.E[-1])
            else:
                self.K[i,i] = abs(self.K[i,i-1]) + stiffness(self.l[i],self.A[i],self.E[i])
    
    def Force(self,f,ele):
        '''
        Defines the forces acting on the nodes. 
        '''
        self.F=zeros(self.n_nodes)
        for i in range(len(ele)):
            self.F[ele[i]-1] = f[i]
    
    def temp_effects(self,alpha,delta_T):
        '''
        Calculates the force due to temperature change of the environment. Uses thermal coefficient of expansion and change in temperature.
        '''
        def element_temp_force(a,delta_T,E,A):
            return E*A*a*delta_T*array([-1,1])
        
        self.alpha = alpha
        self.delta_T = delta_T
        self.temp_force = zeros(self.n_nodes)
        for i in range(self.n_elements):
            self.temp_force[i] += element_temp_force(self.alpha[i],self.delta_T,self.E[i],self.A[i])[0]
            self.temp_force[i+1] += element_temp_force(self.alpha[i],self.delta_T,self.E[i],self.A[i])[1]
        self.F += self.temp_force
    
    def boundary_conditions(self,fixed,specified_displacement=False): 
        ''' 
        Applies displacement boundary conditions. Returns a displacement matrix with unknown locations as -99999. Size - (number of nodes x 1)
        '''
        self.Q = ones(self.n_nodes)
        self.Q.fill(-99999)
        if specified_displacement is not False:
            self.Q[int(specified_displacement[1])-1] = specified_displacement[0] # specified_displacement is an array - [value of displacement,position]
        if type(fixed)==int:
            self.Q[fixed-1] = 0
        else:
            for i in range(len(fixed)):
                self.Q[fixed[i]-1] = 0
                

    def solve_mat(self,F,k):
        return linalg.solve(k,F) # matrix solver
    
    def penalty_method(self):
        ''' 
        First method to solve the matrix equations without using inverse of a matrix
        For more information refer - http://web.iitd.ac.in/~hegde/fem/lecture/lecture9.pdf 
        '''
        C = abs(amax(self.K))*10**4
        ind = where(self.Q!=-99999)
        k_copy = self.K.copy()
        for i in range(self.n_nodes):
            if self.Q[i]!=-99999:
                k_copy[i,i] += C
                self.F[i] += C*self.Q[i]
        self.p_sol = self.solve_mat(self.F,k_copy)
        self.reac_forces1 = -C*(self.p_sol[ind] - self.Q[ind])
        self.Q = self.p_sol 
        
    
    def elimination(self):
        ''' 
        Elimination method to solve the matrix without using inverse. Genereally can work on smaller problems with fewer nodes
        '''
        ind1 = where(self.Q==-99999) # indices where we have to find the new values
        ind2 = where(self.Q!=-99999)# indices where the boundary conditions are known
        new_k = delete(self.K,ind2,axis=0)
        new_k = delete(new_k,ind2,axis=1)
        new_f = zeros(len(ind1[0]))
        for i in range(self.Q.shape[0]):
            if i in ind1[0]:
                new_f[i-1] = self.F[i] - dot(self.K[i,ind2],self.Q[ind2])[0]
        print(new_f)
        self.e_sol = self.solve_mat(new_f,new_k)
        self.Q[ind1] = self.e_sol
        self.reac_forces2 = array([dot(self.K[i],self.Q)-self.F[i] for i in ind2[0]]) # Reaction forces at fixed ends
        
    def stress_strain(self):
        '''
        Defines the stress strain relationship using Hooke's Law and calculates stress and strain for each element.
        '''
        self.strains = zeros(self.n_elements)
        self.stresses = zeros(self.n_elements)
        for i in range(self.n_elements):
            if self.temperature_effect:
                self.strains[i] = ((self.Q[i+1]-self.Q[i])/self.l[i]) - self.aplha[i]*self.delta_T
                self.stresses[i] = self.strains[i]*self.E[i]
            else:
                self.strains[i] = (self.Q[i+1]-self.Q[i])/self.l[i]
                self.stresses[i] = self.strains[i]*self.E[i] 
    
    def solve(self, method):
        if method =='elimination':
            self.elimination()
            self.stress_strain()
            print("Reaction forces -- ",self.reac_forces2)
        elif method=='penalty':
            self.penalty_method()
            self.stress_strain()
            print("Reaction forces -- ",self.reac_forces1)
        else:
            print("Error: No method of type ",method," found")
    
    def print_func(self):
        print("Total Force -- ",self.F)
        print("Displacement -- ", self.Q)
        print("Global element stiffness -- \n",self.K)
        print("strains -- ", self.strains)
        print("Stresses -- ", self.stresses)
        
if __name__ =="__main__":        
    #################################
    ######## Characteristics ########
    E = (10**9)*array([200.0,74.0,68.0])
    A = (10**-2)*array([1,0.75,0.5])
    l = array([1,0.75,0.5])
    n_elements = 3
    method = 'elimination'
    #################################
    ########### Forces ##############
    F = array([2000,1000])
    F_position = array([4,2]) # The node at which the Force is acting
    #################################
    ####### Temperature effects #####
    temperature_effect = True
    alpha = 10**-6*array([12,20])    # Thermal expansion coefficient. Unit is per Deg Celsius
    delta_T = 50 # deg Celsius
    #################################
    #### Boundary conditions ########
    fixed_pos = 1
    node_with_displacement = False # node_with_displacement is an array - [value of displacement,position]

    bar = Structural1D(E,A,l,n_elements,temperature_effect)
    bar.find_stiffness_matrix()
    bar.Force(F,F_position)
    bar.temp_effects(alpha,delta_T)
    bar.boundary_conditions(fixed_pos,node_with_displacement)
    bar.solve(method)
    bar.print_func()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
