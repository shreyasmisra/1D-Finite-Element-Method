from numpy import *
import matplotlib.pyplot as plt

class FluidFlow:
    def __init__(self,K_p,A,l,n_elements):
        self.K_p = K_p
        self.l = l
        self.A = A
        self.n_elements = n_elements
        self.n_nodes = n_elements+1
        self.L = self.l[0]*self.n_elements

    
    def find_stiffness_matrix(self):
        '''
        Calculates the Global stiffness matrix based on Minimum Potential Energy Principle. 
        K is the global stiffness matrix of shape - number of nodes x number of nodes. Assuming 1 DoF at each node
        '''
        def stiffness(l,A):
                return (self.K_p*A)/l # array([[1,-1],[-1,1]]). Element stiffness matrix as given by Minimum Potential Energy Method
        self.K = zeros((self.n_nodes,self.n_nodes)) # Global stiffness matrix
        for i in range(self.n_elements):
            self.K[i,i+1] = -1*stiffness(self.l[i],self.A[i])
            self.K[i+1,i] = -1*stiffness(self.l[i],self.A[i])
            if i == 0:
                self.K[0,0] = stiffness(self.l[0],self.A[0])
                self.K[-1,-1] = stiffness(self.l[-1],self.A[-1])
            else:
                self.K[i,i] = abs(self.K[i,i-1]) + stiffness(self.l[i],self.A[i])
    
    
    def internal_sinks(self,Q,A,l):
        ''' 
        Calculates the flow rates from internal sources and sinks. For example Pumps
        '''
        return (Q*A*l/2)*array([1,1])
        
    def surface_flow(self,q,t,l):
        ''' 
        Calculates surface-edge flow rates, such as from a river or stream 
        '''
        return (q*t*l/2)*array([1,1]).T
    
    def total_flow(self,Q,q,t):
        ''' 
        Calculates the total flow in the system at each node
        '''
        self.Q_force = zeros(self.n_nodes)
        for i in range(self.n_elements):
            Q0 = self.surface_flow(q,t,self.l[i])
            Q1 = self.internal_sinks(Q,self.A[i],self.l[i])
            self.Q_force[i] += Q0[0]+Q1[0]
            self.Q_force[i+1] += Q0[1]+Q1[1]
            
        
    def boundary_conditions(self,value): # from higher to lower
        ''' 
        Applies the Head Boundary condition at the nodes where Head is specified. Usually the inlet and outlet nodes
        '''
        self.H = ones(self.n_nodes)
        self.H.fill(-99999) # random value -99999 indicates the unknown values
        self.H[0] = value[0] # First value specified is for Inlet
        self.H[-1] = value[-1] # For outlet
    
    
    def solve_mat(self,F,k):
        return linalg.solve(k,F) # matrix solver
    
    def penalty_method(self):
        ''' 
        First method to solve the matrix equations without using inverse of a matrix
        For more information refer - http://web.iitd.ac.in/~hegde/fem/lecture/lecture9.pdf 
        '''
        C = abs(amax(self.K))*10**4
        k_copy = self.K.copy()
        for i in range(self.n_nodes):
            if self.H[i]!=-99999:
                k_copy[i,i] += C
                self.Q_force[i] += C*self.H[i]
        self.p_sol = self.solve_mat(self.Q_force,k_copy)
        self.H = self.p_sol # Final solution fo pressure head along the flow path
    
    def elimination(self):
        ''' 
        Elimination method to solve the matrix without using inverse. Genereally can work on smaller problems with fewer nodes
        '''
        ind1 = where(self.H==-99999) # indices where we have to find the new values
        ind2 = where(self.H!=-99999)# indices where the boundary conditions are known
        new_k = delete(self.K,ind2,0)
        new_k = delete(new_k,ind2,1)
        new_f = zeros(len(ind1[0]))
        for i in range(self.H.shape[0]):
            if i in ind1[0]:
                new_f[i-1] = self.Q_force[i] - dot(self.K[i,ind2],self.H[ind2])[0]
        self.e_sol = self.solve_mat(new_f,new_k)
        self.H[ind1] = self.e_sol # final solution using elimination method
        
        
    def velocity_distribution(self):
        ''' 
        Calculates the Velocity profile for all elements
        '''
        self.velocity = zeros(self.n_elements)
        for i in range(self.n_elements):
            self.velocity[i] = -self.K_p *(self.H[i+1]-self.H[i])/self.l[i]
    
    
    def flow_rate(self):
        ''' 
        Flow rate calculation
        '''
        self.flow = zeros(self.n_elements)
        for i in range(self.velocity.shape[0]):
            self.flow[i] = self.velocity[i]*self.A[i]
      
    
    def solve(self, method):
        if method =='elimination':
            self.elimination()
            self.velocity_distribution()
            self.flow_rate()
        elif method=='penalty':
            self.penalty_method()
            self.velocity_distribution()
            self.flow_rate()
        else:
            print("Error: No method of type ",method," found")
    
    
    def print_func(self):
        print("Total Flow -- ",self.Q_force)
        print("Heads -- ", self.H)
        print("Global element stiffness -- \n",self.K)
        print("Velocity distribution -- ", self.velocity)
        print("FLow Rate -- ", self.flow)
        
        
    def visualize(self):
        ''' 
        Visualize the change in the flow rate, velocity and head over the domain
        '''
        lengths = linspace(0,self.L,self.n_nodes)
        self.velocity = insert(self.velocity,0,0)
        self.flow = insert(self.flow,0,0)
        plt.plot(lengths,self.velocity,'b',label="Velocity")
        plt.plot(lengths,self.H,'k',label="Heads")
        plt.title("Variation of Head and Velocity over the domain")
        plt.legend(loc='upper right')

#### SAMPLE PROGRAM UTILIZATION ########################
if __name__=="__main__":  
    ####### Fluid domain ##############
    n_elements = 15 # number of elements to discretize domain
    K_p = 1 # Permeability coefficient. Based on viscosity of the fluid. Unit - cm/s
    A = array([3,2,2,3,3,2,1,5,4,6,6,1,2,6,2]) # Area of each of the element
    L = [1]*n_elements # length of each element. Can also be different. Specify similar to Area array
    solution_method = 'elimination'
    #######################################
    ####### Sources/Sinks and Surface flows ################
    Q = 200 # sources/sinks
    q = 0 # surface flows
    t = 0 # thickness of the surface flow
    ######################################
    #### Boundary Conditions #############
    H = (50,-4) # Heads at the ends
    
    fluid = FluidFlow(K_p,A,L,n_elements)
    fluid.find_stiffness_matrix()
    fluid.total_flow(Q, q, t)
    fluid.boundary_conditions(H)
    fluid.solve(solution_method)
    fluid.print_func()
    fluid.visualize()





