# SCM

| OS        |  Status |
| :-------------: |:-------------:|
| Linux       | ![L Py 3.7 - 3.9](https://github.com/maichmueller/scm/workflows/L%20Py%203.7%20-%203.9/badge.svg)    |
| Windows | ![W Py 3.7 - 3.9](https://github.com/maichmueller/scm/workflows/W%20Py%203.7%20-%203.9/badge.svg) | 
| Mac | ![M Py 3.7 - 3.9](https://github.com/maichmueller/scm/workflows/M%20Py%203.7%20-%203.9/badge.svg) |

A Python package implementing Structural Causal Models (SCM).

The library uses the CAS library [SymPy](https://github.com/sympy/sympy) to allow the user to state arbitrary assignment functions and noise distributions as supported by SymPy and builds the DAG with [networkx](https://github.com/networkx/networkx).

It supports the features:
  - Sampling
  - Intervening
  - Plotting
  - Printing
  
 and by extension all methods on a DAG provided by networkx after accessing the member variable dag
  
# Installation
Git clone the repository and run the setup.py file
```
git clone https://github.com/maichmueller/scm
cd scm
python setup.py install
```

# Example usage

To build the DAG

![X \rightarrow Y \leftarrow Z \rightarrow X](https://latex.codecogs.com/svg.latex?&space;X{\rightarrow}{Y}{\leftarrow}{Z}{\rightarrow}X)


with the assignments

![Z = LogLogistic(alpha=1, beta=1)](https://latex.codecogs.com/svg.latex?&space;Z=\text{LogLogistic}(\alpha=1,\beta=1))

![X = 3Z^2{\cdot}N](https://latex.codecogs.com/svg.latex?&space;X={3Z^2}{\cdot}N\quad[N=\text{LogNormal}(\mu=1,\sigma=1)])

![Y = 2Z + \sqrt{X} + N](https://latex.codecogs.com/svg.latex?&space;Y=2Z+\sqrt{X}+N\quad[N=\text{Normal}(\mu=2,\sigma=1)])

one can describe the assignments as strings

```python
from scm import SCM

myscm = SCM(
    [
        "Z = N, N ~ LogLogistic(alpha=1, beta=1)", 
        "X = N * 3 * Z ** 2, N ~ LogNormal(mean=1, std=1)", 
        "Y = N + 2 * Z + sqrt(X), N ~ Normal(mean=2, std=1)"
    ]   
)
```

or build the assignments piecewise themselves via an assignment map

```python
from scm import SCM
from sympy.stats import LogLogistic, LogNormal, Normal


functional_map = {
   "Z": (
       "N",
       LogLogistic("N", alpha=1, beta=1)
   ),
   "X": (
       "N * 3 * Z ** 2",
       LogNormal("N", mean=1, std=1),
   ),
   "Y": (
       "N + 2 * Z + sqrt(X)",
       Normal("N", mean=2, std=1),
   ),
}

myscm = SCM(functional_map)
```
You can print the current status of your SCM
```python
print(myscm)
```
```
Structural Causal Model of 3 variables: Z, X, Y
Following variables are actively intervened on: []
Current Assignments are:
Z := f(N) = N	 [ N := LogLogistic(1, 1) ]
X := f(N, Z) = N * 3 * Z ** 2	 [ N := LogNormal(1, 1) ]
Y := f(N, Z, X) = N + 2 * Z + sqrt(X)	 [ N := Normal(2, 1) ]
```
Perform the Do-intervention ![\text{do}(X=1=)](https://latex.codecogs.com/svg.latex?&space;\text{do}(X=1)) :
```python
myscm.do_intervention(["X"], [1])
```
and sample from it
```python
myscm.sample(5)
```
```
   X          Z          Y
0  1   0.020325   0.993296
1  1   0.561370   4.014003
2  1   0.047893   3.856194
3  1  26.286726  55.666348
4  1   1.870452   6.657642
```
If you have graphviz installed, you can also use it to plot the graph in an easy manner
```python
import matplotlib.pyplot as plt

myscm.plot(node_size=1000, alpha=1)
plt.show()
```
![Plot example](docs/images/plot.png)
