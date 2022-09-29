import random
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time
import operator

# ax = plt.axes(projection = "3d")
#------------------------------------------------------------------------------
optimal = 999
activatePlot = True


def objective_function(x):
# PROBLEMS    
    # Beale Function
    # y = (1.5-x[0]+x[0]*x[1])**2+(2.25-x[0]*x[0]*x[1]**2)**2+(2.625-x[0]+x[0]*x[1]**3)**2
    
    # Ackley Function
    # y = -20*math.exp(-0.2*math.sqrt(0.5*(x[0]**2+x[1]**2)))-math.exp(0.5*(math.cos(2*math.pi*x[0])+math.cos(2*math.pi*x[1])))+math.e+20

    # Easom Function
    # y = -math.cos(x[0])*math.cos(x[1])*math.exp(-((x[0]-math.pi)**2+(x[1]-m ath.pi)**2))
    
    # Levi Function N.13
    # y = math.sin(3*math.pi*x[0])**2+(x[0]-1)**2+(1+math.sin(3*math.pi*x[1])**2)+(x[1]-1)**2*(1+math.sin(2*math.pi*x[1])**2)

    # Contoh Soal 1 (MAX)
    # y = -(x[0]**2+x[1]**2)+4
    
    # Contoh Soal 2 (MIN)
    # y = (2-x[0])**2+40*(x[1]-x[0]**2)**2
    
    # Contoh Soal 3 (MIN)
    # y = 3*x[0]**2-2*x[0]*x[1]+3*x[1]**2-x[0]-x[1]
    
    # Sphere Function
    # y = x[0]**2+x[1]**2+x[2]**2+x[3]**2+x[4]**2
    
    # Goldstein-Price Function
    # y = (1+(x[0]+x[1]+1)**2*(19-14*x[0]+3*x[0]**2-14*x[1]+6*x[0]*x[1]+3*x[1]**2))*(30+(2*x[0]-3*x[1])**2*(18-32*x[0]+12*x[0]**2+48*x[1]-36*x[0]*x[1]+27*x[1]**2))
    
    # Booth Function
    # y = (x[0]+2*x[1]-7)**2+(2*x[0]+x[1]-5)**2
    
    # Matyas Function
    # y = 0.26*(x[0]**2+x[1]**2)-0.48*x[0]*x[1]
    
    # Himmelblau's Function
    y = (x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2
    
    # Three-hump Camel Function
    # y = 2*x[0]**2-1.05*x[0]**4+(x[0]**6)/6+x[0]*x[1]+x[1]**2
    
    # Rastrigin Function
    # y = 10*4+(x[0]**2-10*math.cos(2*math.pi*x[0]))+(x[1]**2-10*math.cos(2*math.pi*x[1]))+(x[2]**2-10*math.cos(2*math.pi*x[2]))+(x[3]**2-10*math.cos(2*math.pi*x[3]))
    
    # Rosenbrock Function
    # y = (100*(x[1]-x[0]**2)**2+(1-x[0])**2)+(100*(x[2]-x[1]**2)**2+(1-x[1])**2)+(100*(x[3]-x[2]**2)**2+(1-x[2])**2)
    
    # McCormick Function
    # y = math.sin(x[0]+x[1])+(x[0]-x[1])**2-1.5*x[0]+2.5*x[1]+1
    
    # Bukin Function
    # y = 100*math.sqrt(abs(x[1]-0.01*(x[0])**2))+0.01*abs(x[0]+10)

    # Cross in tray Function
    # y = -0.0001*(abs(math.sin(x[0])*math.sin(x[1])*math.exp(abs(100-(math.sqrt(x[0]**2+x[1]**2))/math.pi))+1))**0.1
    
    # Eggholder Function
    # y = (x[1]+47)*math.sin(math.sqrt(abs(x[0]/2+(x[1]+47))))-x[0]*math.sin(math.sqrt(abs(x[0]-(x[1]+47))))
    
    # Holder table Function
    # y = -abs(math.sin(x[0])*math.cos(x[1])*math.exp(abs(1-(math.sqrt(x[0]**2+x[1]**2))/math.pi)))
    
    # Schaffer Function N.2
    # y = 0.5+((math.sin(x[0]**2-x[1]**2))**2-0.5)/((1+0.001*(x[0]**2+x[1]**2))**2)
   
    # Schaffer Function N.4
    # y = 0.5+((math.cos(math.sin(abs(x[0]**2-x[1]**2))))**2-0.5)/((1+0.001*(x[0]**2+x[1]**2))**2)
    
    # Styblinski–Tang Function
    # y = ((x[0]**4-16*x[0]**2+5*x[0])+(x[1]**4-16*x[1]**2+5*x[1])+(x[2]**4-16*x[2]**2+5*x[2])+(x[3]**4-16*x[3]**2+5*x[3]))/2

    # Test Case 1 (MAX)
    # y = (1-x[0])**2+100*(x[1]-x[0]**2)**2
    
    # Test Case 2 (MIN)
    # y = (x[0]-50)**2+(x[1]-50)**2
    
    # Test Case 3 (MAX)
    # y = (1+math.cos(12*math.sqrt(x[0]**2+x[1]**2)))/(0.5*(x[0]**2+x[1]**2)+2)
    
    # Test Case 4 (MIN)
    # y = (4-2.1*x[0]**2+(1/3)*x[0]**4)*x[0]**2+x[0]*x[1]+4*(x[1]**2-1)*x[1]**2
    
    # Test Case 5 (MIN)
    # y = 2+(x[0]-2)**2+(x[1]-1)**2
    
    # Test Case 6 (MIN)
    # y = (1/(x[0]**2+x[1]**2+1))-1.1*math.exp(-(x[0]**2+x[1]**2))
    
    # Test Case 7 (MAX)
    # y = (x[0]+2*x[1]-7)**2+(2*x[0]+x[1]-5)**2
    
    # Test Case 8 (MAX)
    # y = (x[0]+2*x[1]-2)**3+(x[0]*x[1])**2
    
    # Test Case 9 (MAX)
    # y = (x[0]+8*x[1]-2)**4+(4*x[0]*x[1])**3+(2*x[0]-x[1])**2
    
    # Test Case 10 (MAX)
    # y = 1+((x[0]+x[1]+1)**2)*(19-14*x[0]+3*x[0]**2-14*x[1]+6*x[0]*x[1]+3*x[1]**2)
    
    return y

objective_function_vec = np.vectorize(objective_function)

# Optimal Value and Upper and lower bounds of variables

# Beale Function
# bounds=[(-4.5,4.5),(-4.5,4.5)]
# optimal = 0

# Ackley Function
# bounds=[(-5,5),(-5,5)]
# optimal = 0

# Easom Function 
# bounds=[(-100,100),(-100,100)]
# optimal = -1

# Levi Function N.13
# bounds=[(-10,10),(-10,10)]
# optimal = 0

# Contoh Soal 1 (MAX)
# bounds=[(-3,3),(-3,3)]

# Contoh Soal 2 (MIN)
# bounds=[(-10,10),(-10,10)]

# Contoh Soal 3 (MIN)
# bounds= [(-5,5),(-5,5)]

# Sphere Function
# bounds=[(-999999,999999),(-999999,999999),(-999999,999999),(-999999,999999),(-999999,999999)]
# optimal = 0

# Goldstein-Price Function
# bounds=[(-2,2),(-2,2)]
# optimal = 3

# Booth Function
# bounds=[(-10,10),(-10,10)]
# optimal = 0

# Matyas Function
# bounds=[(-10,10),(-10,10)]
# optimal = 0

# Himmelblau's Function
bounds= [(-5,5),(-5,5)]
optimal = 0

# Three-hump Camel Function
# bounds= [(-5,5),(-5,5)]
# optimal = 0

# Rastrigin Function
# bounds = [(-5.12,5.12),(-5.12,5.12),(-5.12,5.12),(-5.12,5.12)]
# optimal = 0

# Rosenbrock Function
# bounds=[(-1000,1000),(-1000,1000),(-1000,1000),(-1000,1000)]
# optimal = 0

# McCormick Function
# bounds=[(-1.5,4),(-3,4)]
# optimal = -1.9133

# Bukin Function
# bounds=[(-15,-5),(-3,3)]
# optimal = 0

# Cross in tray Function
# bounds=[(-10,10),(-10,10)]
# optimal = -2.06261
    
# Eggholder Function
# bounds=[(-512,512),(-512,512)]
# optimal = -959.6407
   
# Holder table Function
# bounds=[(-10,10),(-10,10)]
# optimal = -19.2085

# Schaffer Function N. 2
# bounds=[(-100,100),(-100,100)]
# optimal = 0
   
# Schaffer Function N. 4
# bounds=[(-100,100),(-100,100)]
# optimal = 0.292579
  
# Styblinski–Tang Function
# bounds=[(-5,5),(-5,5),(-5,5),(-5,5)]
# optimal = -39.16599*4

# Test Case 1 (MAX)
# bounds = [(-0.5,2.5),(-0.5,2.5)]

# Test Case 2 (MIN)
# bounds = [(1,100),(1,100)]

# Test Case 3 (MAX)
# bounds = [(-2,2),(-2,2)]

# Test Case 4 (MIN)
# bounds = [(0,50),(0,50)]

# Test Case 5 (MIN)
# bounds = [(-20,20),(-20,20)]

# Test Case 6 (MIN)
# bounds = [(-3,3),(-3,3)]

# Test Case 7 (MAX)
# bounds = [(-10,10),(-10,10)]

# Test Case 8 (MAX)
# bounds = [(-5,5),(-5,5)]

# Test Case 9 (MAX)
# bounds = [(-20,20),(-20,20)]

# Test Case 10 (MAX)
# bounds = [(-5,5),(-5,5)]


nv = 2                   # number of variables
mm = -1                   # if minimization problem, mm = -1; if maximization problem, mm = 1

# THE FOLLOWING PARAMETERS ARE OPTINAL.
# Global
particle_size=10        # number of particles
iterations=100           # max number of iterations
# Particle Swarm Optimization
w=0.85                    # inertia constant
c1=1                    # cognative constant
c2=2                    # social constant
# Simulated Annealing
initial_temperature = 100
cooling = 0.8  # cooling coefficient
no_attempts = 100 # number of attempts in each level of temperature
# Stopping Criteria
useOptimal = True # stop iterations if optimality is found
# END OF THE CUSTOMIZATION SECTION
#------------------------------------------------------------------------------   
ops = {
       -1: operator.lt,
       1: operator.gt
       }
op_func = ops[mm]

class Particle:
    def __init__(self,bounds):
        self.particle_position = []                     # particle position
        self.particle_velocity = []                     # particle velocity
        self.local_best_particle_position = []          # best position of the particle
        self.particle_temperature = initial_temperature # current temperature of particle
        self.n = 1 # Number of solutions accepted
        self.accept = True # Accept new solution or not
        self.fitness_local_best_particle_position = initial_fitness  # initial objective function value of the best particle position
        self.fitness_particle_position=initial_fitness             # objective function value of the particle position
 
        for i in range(nv):
            self.particle_position.append(random.uniform(bounds[i][0],bounds[i][1])) # generate random initial position
            self.particle_velocity.append(random.uniform(-1,1)) # generate random initial velocity
 
        self.fitness_particle_position=objective_function(self.particle_position)
        self.E = abs(self.fitness_particle_position - self.fitness_local_best_particle_position)
        self.EA = self.E
    
    def evaluate(self, objective_function, current_temperature):
        self.fitness_particle_position=objective_function(self.particle_position)
        self.E = abs(self.fitness_particle_position - self.fitness_local_best_particle_position)
        if op_func(self.fitness_particle_position, self.fitness_local_best_particle_position):
            self.accept = True
        else:
            p = math.exp(-self.E/(self.EA*current_temperature))
            # make a decision to accept the worse solution or not
            if random.random()<p:
                self.accept = True # this worse solution is accepted
            else:
                self.accept = False # this worse solution is not accepted
        if self.accept:
            self.local_best_particle_position = self.particle_position                  # update the local best
            self.fitness_local_best_particle_position = self.fitness_particle_position  # update the fitness of the local best    
            self.n += 1 # count the solutions accepted
            self.EA = (self.EA *(self.n - 1) + self.E)/self.n # update EA
                
    def update_velocity(self,global_best_particle_position):
        for i in range(nv):
            r1=random.random()
            r2=random.random()
 
            cognitive_velocity = c1*r1*(self.local_best_particle_position[i] - self.particle_position[i])
            social_velocity = c2*r2*(global_best_particle_position[i] - self.particle_position[i])
            self.particle_velocity[i] = w*self.particle_velocity[i]+ cognitive_velocity + social_velocity
            
 
    def update_position(self,bounds):
        for i in range(nv):
            self.particle_position[i]=self.particle_position[i]+self.particle_velocity[i]
 
            # check and repair to satisfy the upper bounds
            if self.particle_position[i]>bounds[i][1]:
                self.particle_position[i]=bounds[i][1]
            # check and repair to satisfy the lower bounds
            if self.particle_position[i] < bounds[i][0]:
                self.particle_position[i]=bounds[i][0]
                 
class PSO():
    def __init__(self,objective_function,bounds,particle_size,iterations):
 
        fitness_global_best_particle_position=initial_fitness
        global_best_particle_position=[]
 
        swarm_particle=[]
        for i in range(particle_size):
            swarm_particle.append(Particle(bounds))
        current_temperature = initial_temperature
        A=[]
        B=[]
        timeTaken = 0
        resultOptimal = False
        for i in range(iterations):
            startTime = time.time()
            iteration_num = i + 1
            plotters_x = []
            plotters_y = []
            plotters_z = []
            B.append([])
            for j in range(no_attempts):
                for k in range(particle_size):
                    swarm_particle[k].evaluate(objective_function, current_temperature)
     
                    if op_func(swarm_particle[k].fitness_particle_position, fitness_global_best_particle_position):
                        global_best_particle_position = list(swarm_particle[k].particle_position)
                        fitness_global_best_particle_position = float(swarm_particle[k].fitness_particle_position)
                    
                for k in range(particle_size):
                    swarm_particle[k].update_velocity(global_best_particle_position)
                    swarm_particle[k].update_position(bounds)
            current_temperature = current_temperature*cooling
                
            timeTaken += time.time() - startTime   
            
            for j in range(particle_size):
                plotters_x.append(swarm_particle[j].particle_position[0])
                plotters_y.append(swarm_particle[j].particle_position[1])
            
            A.append(fitness_global_best_particle_position) # record the best fitness
            for j in range(particle_size):
                B[i].append(swarm_particle[j].fitness_particle_position)
            
            # if (i + 1) % 50 == 0 or i == 0:
            if ((i + 1) % 1 == 0 or i == 0 or abs(A[iteration_num - 1] - optimal) <= 0) and activatePlot:
            # if (i + 1) % 1 == 0 or i == 0 and activatePlot:
            # if activatePlot:
                plt.xlim(bounds[0])
                plt.ylim(bounds[1])
                
                for j in range(len(plotters_x)):
                    plotters_z.append(objective_function([plotters_x[j],plotters_y[j]]))
                
                # Plotting the Function
                x = np.linspace(bounds[0][0],bounds[0][1], 100)
                y = np.linspace(bounds[1][0],bounds[1][1], 100)
                
                X, Y = np.meshgrid(x, y)
                Z = [[] for Null in range(len(X))]
                for i in range(len(X)):
                    myList = []
                    for j in range(len(X)):    
                        myList.append(objective_function([X[i][j],Y[i][j]]))
                    Z[i] = myList
                
                
                ax = plt.axes(projection = "3d")
                ax.view_init(80, 30)
                ax.contour3D(X, Y, Z, 50, proj_type='ortho', alpha = 0.3)
                
                # Plotting the swarm
                ax.scatter3D(plotters_x, plotters_y, plotters_z, color='red', alpha=1, marker='.')
                ax.set_title("PSO-SA (Himmelblau's Function\nIteration %i" % iteration_num)
                plt.show()
            
            # Stopping criteria
            if iteration_num > 1 and useOptimal:
                if abs(A[iteration_num - 1] - optimal) <= 0:
                    print('Stopped at iteration no.', iteration_num)
                    resultOptimal = True
                    break
            
        if resultOptimal:
            print('Result is globally optimal')
        else:
            print('Stopped at iteration no.', iterations)
            if useOptimal:
                print('Result is not globally optimal, with a delta of: ', abs(fitness_global_best_particle_position - optimal))
            
        print('Optimal solution:', global_best_particle_position)
        print('Objective function value:', fitness_global_best_particle_position)
        print('Time taken: ', timeTaken, ' seconds')
        if nv > 2:
            print('Graph not drawn because dimension is > 2')
        plt.plot(B, "blue", alpha = 0.1)
        plt.plot(A, "red", alpha = 0.8, label = 'gbest')
        plt.ylabel('Objective Value')
        plt.xlabel('Iteration')
        plt.title("PSO-SA Objective Value/Iteration\n(Himmelblau's Function)")
        plt.legend(['GBest', 'Ofv of Particles'])
        ax1 = plt.gca()
        leg = ax1.get_legend()
        leg.legendHandles[0].set_color('red')
        leg.legendHandles[0].set_alpha(1)
        leg.legendHandles[1].set_color('blue')
        leg.legendHandles[1].set_alpha(1)
        plt.show()
#------------------------------------------------------------------------------
if mm == -1:
    initial_fitness = float("inf") # for minimization problem
if mm == 1:
    initial_fitness = -float("inf") # for maximization problem
#------------------------------------------------------------------------------   
# Main 
         
PSO(objective_function,bounds,particle_size,iterations)
