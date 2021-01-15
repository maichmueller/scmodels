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
Either install via pip
```
pip install scmodels
```
or via cloning the repository and running the setup.py file
```
git clone https://github.com/maichmueller/scm
cd scm
python setup.py install
```

# Building an SCM

To build the DAG

![X \rightarrow Y \leftarrow Z \rightarrow X](https://latex.codecogs.com/svg.latex?&space;X{\rightarrow}{Y}{\leftarrow}{Z}{\rightarrow}X)


with the assignments

![Z ~ LogLogistic(alpha=1, beta=1)](https://latex.codecogs.com/svg.latex?&space;Z\sim\text{LogLogistic}(\alpha=1,\beta=1)])

![X = 3Z^2{\cdot}N](https://latex.codecogs.com/svg.latex?&space;X={3Z^2}{\cdot}N\quad[N=\text{LogNormal}(\mu=1,\sigma=1)])

![Y = 2Z + \sqrt{X} + N](https://latex.codecogs.com/svg.latex?&space;Y=2Z+\sqrt{X}+N\quad[N=\text{Normal}(\mu=2,\sigma=1)])

There are 3 different ways of declaring the SCM:

## 1. List Of Strings
Describe the assignments as strings of the form:

'VAR = FUNC(Noise, parent1, parent2, ...), Noise ~ DistributionXYZ'

Note that - out of convenience - in this case, one does not need to (and isn't allowed to)
restate the noise symbol string in the distribution (as would otherwise be necessary
in constructing sympy distributions).


```python
from scmodels import SCM

assignment_seq = [
    "Z = M, M ~ LogLogistic(alpha=1, beta=1)",
    "X = N * 3 * Z ** 2, N ~ LogNormal(mean=1, std=1)",
    "Y = P + 2 * Z + sqrt(X), P ~ Normal(mean=2, std=1)"
]

myscm = SCM(assignment_seq)
```

Agreements:
- The name of the noise variable in the distribution specification
(e.g. `P ~ Normal(mean=2, std=1)`) has to align with the noise variable
name (`P`) of the assignment string.

## 2. Assignment Map

One can construct the SCM via an assignment map with the variables as keys
and a tuple defining the assignment and the noise.

### 2-Tuple: Assignments via SymPy parsing

To refer to SymPy's string parsing capability (this includes numpy functions) provide a dict entry
with 2-tuple of the form:

`'var': ('assignment string', noise)`




```python
from sympy.stats import LogLogistic, LogNormal, Normal


assignment_map = {
   "Z": (
       "M",
       LogLogistic("M", alpha=1, beta=1)
   ),
   "X": (
       "N * 3 * Z ** 2",
       LogNormal("N", mean=1, std=1),
   ),
   "Y": (
       "P + 2 * Z + sqrt(X)",
       Normal("P", mean=2, std=1),
   ),
}

myscm2 = SCM(assignment_map)
```

Agreements:
- the name of the noise distribution provided in its constructor
(e.g. `Normal("N", mean=2, std=1)`) has to align with the noise variable
name (`N`) of the assignment string.

### 3-Tuple: Assignments with arbitrary callables

One can also declare the SCM via specifying the variable assignment in a dictionary with the
variables as keys and as values a sequence of length 3 of the form:

`'var': (['parent1', 'parent2', ...], Callable, Noise)`

This allows the user to supply complex functions outside the space of predefined functions.


```python
import numpy as np


def Y_assignment(p, z, x):
    return p + 2 * z + np.sqrt(x)


functional_map = {
   "Z": (
       [],
       lambda m: m,
       LogLogistic("M", alpha=1, beta=1)
   ),
   "X": (
       ["Z"],
       lambda n, z: n * 3 * z ** 2,
       LogNormal("N", mean=1, std=1),
   ),
   "Y": (
       ["Z", "X"],
       Y_assignment,
       Normal("P", mean=2, std=1),
   ),
}

myscm3 = SCM(functional_map)
```

Agreements:
- The callable's first parameter MUST be the noise input (unless the noise distribution is `None`).
- The order of variables in the parents list determines the semantic order of input for parameters in the functional
(left to right).

# Features

## Prettyprint

The SCM supports a form of informative printing of its current setup,
which includes mentioning active interventions and the assignments.
The function declaration won't be informative if the SCM has been
constructed with custom Callables.


```python
print(myscm)
```

    Structural Causal Model of 3 variables: Z, X, Y
    Variables with active interventions: []
    Assignments:
    Z := f(M) = M	 [ M ~ LogLogistic(alpha=1, beta=1) ]
    X := f(N, Z) = N * 3 * Z ** 2	 [ N ~ LogNormal(mean=1, std=1) ]
    Y := f(P, Z, X) = P + 2 * Z + sqrt(X)	 [ P ~ Normal(mean=2, std=1) ]


In the case of custom callable assignments, the output is less informative


```python
print(myscm3)
```

    Structural Causal Model of 3 variables: Z, X, Y
    Variables with active interventions: []
    Assignments:
    Z := f(M) = __unknown__	 [ M ~ LogLogistic(alpha=1, beta=1) ]
    X := f(N, Z) = __unknown__	 [ N ~ LogNormal(mean=1, std=1) ]
    Y := f(P, Z, X) = __unknown__	 [ P ~ Normal(mean=2, std=1) ]


## Interventions
One can easily perform interventions on the variables,
e.g. a Do-intervention or also general interventions, which remodel the connections, assignments, and noise distributions.
For general interventions, the passing structure is dict of the following form:

    {var: (New Parents (Optional), New Assignment (optional), New Noise (optional))}

Any unchanged part of the original  variable state that is meant to be unchanged is to be passed as `None`.
E.g. to assign a new callable assignment to variable `X` without changing parent or noise, one would call:


```python
my_new_callable = lambda n, z: n + z

myscm.intervention({"X": (None, my_new_callable, None)})
```

For the example of the do-intervention ![\text{do}(X=1=)](https://latex.codecogs.com/svg.latex?&space;\text{do}(X=1)),
one can use the helper method `do_intervention`. The pendant for noise interventions is called `soft_intervention`:


```python
myscm.do_intervention([("X", 1)])

from sympy.stats import FiniteRV

myscm.soft_intervention([("X", FiniteRV(str(myscm["X"].noise), density={-1: .5, 1: .5}))])
```

Calling `undo_intervention` restores the original state of all variables from construction time. One can optionally specify,
which interventions are to be undone by specifying a variable list (by default it undoes all):


```python
myscm.undo_intervention(variables=["X"])
```

## Sampling

The SCM allows drawing as many samples as needed through the method `myscm.sample(n)`.


```python
n = 5
myscm.sample(n)
```

    /home/michael/anaconda3/envs/scm/lib/python3.8/site-packages/sympy/stats/rv.py:1104: UserWarning: 
    The return type of sample has been changed to return an iterator
    object since version 1.7. For more information see
    https://github.com/sympy/sympy/issues/19061
      warnings.warn(filldedent(message))





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Z</th>
      <th>X</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.703978</td>
      <td>175.940459</td>
      <td>25.916327</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.297556</td>
      <td>0.083595</td>
      <td>3.850324</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12.436177</td>
      <td>7677.897504</td>
      <td>113.522429</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.357861</td>
      <td>53.234126</td>
      <td>11.342164</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.221415</td>
      <td>13.729146</td>
      <td>9.006147</td>
    </tr>
  </tbody>
</table>
</div>



If infinite sampling is desired, one can also receive a sampling generator through


```python
container = {var: [] for var in myscm}
sampler = myscm.sample_iter(container)
```

`container` is an optional target dictionary to store the computed samples in.


```python
import pandas as pd

for i in range(n):
    next(sampler)

pd.DataFrame.from_dict(container)
```

    /home/michael/anaconda3/envs/scm/lib/python3.8/site-packages/sympy/stats/rv.py:1104: UserWarning: 
    The return type of sample has been changed to return an iterator
    object since version 1.7. For more information see
    https://github.com/sympy/sympy/issues/19061
      warnings.warn(filldedent(message))





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Z</th>
      <th>X</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.559822</td>
      <td>1.953262</td>
      <td>6.498119</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.660067</td>
      <td>610.427458</td>
      <td>48.137594</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.352375</td>
      <td>1.021604</td>
      <td>2.985338</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.821680</td>
      <td>3.021936</td>
      <td>4.299420</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.304742</td>
      <td>28.855881</td>
      <td>11.735227</td>
    </tr>
  </tbody>
</table>
</div>



If the target container is not provided, the generator returns a new `dict` for every sample.


```python
sample = next(myscm.sample_iter())
pd.DataFrame.from_dict(sample)
```

    /home/michael/anaconda3/envs/scm/lib/python3.8/site-packages/sympy/stats/rv.py:1104: UserWarning: 
    The return type of sample has been changed to return an iterator
    object since version 1.7. For more information see
    https://github.com/sympy/sympy/issues/19061
      warnings.warn(filldedent(message))





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Z</th>
      <th>X</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.435369</td>
      <td>25.468685</td>
      <td>11.333064</td>
    </tr>
  </tbody>
</table>
</div>



## Plotting
If you have graphviz installed, you can plot the DAG by calling

```python
myscm.plot(node_size=1000, alpha=1)
```

![example_plot](https://github.com/maichmueller/scm/blob/master/docs/images/example_plot.png)


```python

```
