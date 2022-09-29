# PSO-SA
 
A hybrid optimization algorithm that combines Particle Swarm Optimization and Simulated Annealing to  solve several test functions.

## How it works

- Uses Simulated Annealing in each iteration of the particle swarm to enable escaping local optimum
- Uses Several test functions and outputs a solution graph

## How to run

1. Choose the test function and bounds to use by uncommenting them
2. Make sure `nv` is aligned with the amount of variables in the problem
3. If `nv > 2`, make sure that `activatePlot = False`
4. Make sure that `mm` is the same as the type of problem (-1 for max, 1 for min)
5. If there is a problem with the optimal value, `useOptimal` can be used
6. Optimization parameters can be freely changed
7. Plot titles need to be manually changed with the chosen test function