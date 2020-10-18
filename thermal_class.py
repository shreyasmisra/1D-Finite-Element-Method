from numpy import *
import matplotlib.pyplot as plt

class Thermal1D:
    def __init__(self,Kt,l,n_elements,h,A=False,Dia=False):
        self.n_elements = n_elements
        self.l = l
        self.L = self.l*self.n_elements
        self.n_nodes = 1+n_elements
        self.Kt = Kt
        self.h = h
        if A:
            self.A = A
            d = array([sqrt(4*self.A[i]/pi) for i in range(self.n_elements)])
            self.P = array([pi*d[i] for i in range(self.n_elements)])
        if Dia and type(Dia)==int or type(Dia)==float:
            self.Dia = Dia
            self.A = (pi/4)*self.Dia**2
            self.P = pi*Dia
            self.A = array([self.A]*self.n_elements)
            self.P = array([self.P]*self.n_elements)
        elif Dia and type(Dia)!=int or type(Dia)!=float:
            self.Dia = Dia
            self.A = array([(pi/4)*self.Dia[i]**2 for i in range(self.n_elements)])
            self.P = array([pi*self.Dia[i] for i in range(self.n_elements)])
    
    def global_stiffness(self,convection_free_end=False):
        '''
        Calculates the Global stiffness matrix K based on convection and conduction.
        '''
        def stiffness_conduction(a,l):
            return (self.Kt*a/l)*array([[1,-1],[-1,1]])
        def stiffness_convection(l,p):
            return (self.h*p*l/6)*array([[2,1],[1,2]])
        def stiffness_convection_right(a):
            return (self.h*a)*array([[0,0],[0,1]])
        def stiffness_convection_left(a):
            return (self.h*a)*array([[1,0],[0,0]])
        
        self.K = zeros((self.n_nodes,self.n_nodes)) # 1 DoF per node
        for i in range(self.n_elements):
            k0 = stiffness_conduction(self.A[i],self.l[i])
            k1 = stiffness_convection(self.l[i],self.P[i])
            kn = k0+k1
            
            self.K[i,i+1] = kn[0,1]
            self.K[i+1,i] = kn[1,0]
            if i==0:
                self.K[0,0] = kn[0,0]
                self.K[-1,-1] = stiffness_conduction(self.A[-1], self.l[-1])[-1,-1]+stiffness_convection(self.l[-1], self.P[-1])[-1,-1]
                if convection_free_end == 'left':
                    self.K[0,0] += stiffness_convection_left(self.A[0])[0,0]
                    # if convection_free_end == 'left':
                    #     self.K[-1,-1] += stiffness_convection_left(self.A[-1])
                elif convection_free_end == 'right':
                    self.K[-1,-1] += stiffness_convection_right(self.A[-1])[-1,-1]
                else:
                    print("Wrong input for convection from end")
            else:
                self.K[i,i] = kn[0,0] + kn[1,1]
                
    def total_heat_force(self,Q,q,T_infinity):
        ''' 
        Calculates the total Heat force F acting on a body including flux on wall surface and cross sectional surface, 
        heat source, convection from end and side surfaces.
        '''
        def heat_source(Q,a,l,x):
            f = zeros(self.n_nodes)
            f[x] = (Q*a*l)/2
            f[x+1] = (Q*a*l)/2
            return f
        def convection_force(l,p,T_infinity,x):
            f = zeros(self.n_nodes)
            f[x] = self.h*p*T_infinity*l/2
            f[x+1] = self.h*p*T_infinity*l/2
            return f
        def heat_flux(q,l,p,x):
            f = zeros(self.n_nodes)
            f[x] = q*p*l/2
            f[x+1] = q*p*l/2
            return f
        def convection_right(a,T_infinity,x):
            f = zeros(self.n_nodes)
            f[x] = 0
            f[x+1] = self.h*T_infinity*a
            return f
        def convection_left(a,T_infinity,x):
            f = zeros(self.n_nodes)
            f[x] = self.h*T_infinity*a
            f[x+1] = 0
            return f
        self.F = zeros(self.n_nodes) # 1 DoF per node
        for i in range(self.n_elements):
            fn = zeros(self.n_nodes)
            f1 = convection_force(self.l[i],self.P[i],T_infinity,i)
            f2 = convection_right(self.A[i],T_infinity,i)
            f3 = convection_left(self.A[i],T_infinity,i)
            f4 = heat_flux(q,self.l[i],self.P[i],i)
            f5 = heat_source(Q,self.A[i], self.l[i],i)
            fn = f1+f2+f3+f4+f5
            
            self.F +=fn
                
    def end_boundary_conditions(self,value): # from higher to lower
        ''' 
        Applies the Temperature Boundary condition at the end nodes where Temperature is specified. Usually the inlet and outlet nodes
        '''
        self.T = ones(self.n_nodes)
        self.T.fill(-99999)
        if type(value)==int:
            self.T[0] = value
        else:
            self.T[0] = value[0]
            self.T[-1] = value[-1]
            
    def solve_mat(self,F,k):
        return linalg.solve(k,F)
    
    def penalty_method(self):
        ''' 
        First method to solve the matrix equations without using inverse of a matrix
        For more information refer - http://web.iitd.ac.in/~hegde/fem/lecture/lecture9.pdf 
        '''
        C = abs(amax(self.K))*10**4
        # self.K[ind,ind] += C
        k_copy = self.K.copy()
        f_copy = self.F.copy()
        for i in range(self.n_nodes):
            if self.T[i]!=-99999:
                k_copy[i,i] += C
                f_copy[i] += C*self.T[i]
        self.p_sol = self.solve_mat(f_copy,k_copy)
        self.T = self.p_sol
        
    def elimination(self):
        ''' 
        Elimination method to solve the matrix without using inverse. Genereally can work on smaller problems with fewer nodes
        '''
        ind1 = where(self.T==-99999) # indices where we have to find the new values
        ind2 = where(self.T!=-99999)# indices where the boundary conditions are known
        new_k = delete(self.K,ind2,0)
        new_k = delete(new_k,ind2,1)
        new_f = zeros(len(ind1[0]))
        for i in range(self.T.shape[0]):
            if i in ind1[0]:
                new_f[i-1] = self.F[i] - dot(self.K[i,ind2],self.T[ind2])[0]
        self.e_sol = self.solve_mat(new_f,new_k)
        self.T[ind1] = self.e_sol  
        
    def heat_transfer_rate(self):
        ''' 
        Calculates the heat transfer rate for all elements
        '''
        self.ht = zeros(self.n_elements)
        for i in range(self.n_elements):
            self.ht[i] = -self.Kt *(self.T[i+1]-self.T[i])/self.l[i]
        
    def solve(self, method):
        if method =='elimination':
            self.elimination()
            self.heat_transfer_rate()
        elif method=='penalty':
            self.penalty_method()
            self.heat_transfer_rate()
        else:
            print("Error: No method of type ",method," found")
    
    def print_func(self):
        print("Total Force -- ",self.F)
        print("Temps -- ", self.T)
        print("Global element stiffness -- \n",self.K)
        print("Heat transfer rate -- ", self.ht)       
                
                
    def visualize(self):
        ''' 
        Visualize the change in the heat transfer rates and temepratures over the domain
        '''
        lengths = linspace(0,self.L,self.n_nodes)
        self.ht = insert(self.ht,0,0)
        plt.plot(lengths,self.ht,'b',label="Heat Transfer rate")
        plt.title("Heat Transfer rate")
        plt.show()
        plt.plot(lengths,self.T,'k',label="Temperature")
        plt.title("Variation of Temperature over the domain")
                        
        
if __name__ =="__main__":                
    ###################################################
    ####### Model characteristics #####################
    n_elements = 10
    thermal_conductivity = 100
    element_length = [0.075]*n_elements
    element_area = [1]*n_elements
    heat_transfer_coefficient = 250 
    d = 0.5 
    T_infinity = 25
    ##################################################
    ####### Forces ##################################
    heat_source = 500
    wall_flux = 100
    end_convection = 'right' # inputs - 'right' or 'left'
    ###########################################
    ####### Boundary Conditions ###############
    T = 1000
    
    thermal = Thermal1D(thermal_conductivity,element_length,n_elements,heat_transfer_coefficient,Dia=d)
    thermal.global_stiffness(end_convection)
    thermal.total_heat_force(heat_source,wall_flux,T_infinity)
    thermal.end_boundary_conditions(T)
    thermal.solve('penalty')
    thermal.print_func()
    thermal.visualize()               
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                    