---
title: 'PyXAB - A Python Library for $\mathcal{X}$-Armed Bandit and Online Blackbox Optimization Algorithms'
tags:
  - Python
  - $\mathcal{X}$-Armed Bandit
  - Online Blackbox Optimization
  - Lipschitz Bandit
authors:
  - name: Wenjie Li
    orcid: 0000-0003-1872-4595
    equal-contrib: true
    affiliation: 1
    corresponding: true # (This is how to denote the corresponding author)
  - name: Haoze Li
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
  - name: Qifan Song
    affiliation: 1
  - name: Jean Honorio
    affiliation: 2
affiliations:
  - name: Department of Statistics, Purdue University, USA
    index: 1
  - name: Department of Computer Science, Purdue University, USA
    index: 2
date: 04 October 2023
bibliography: paper.bib
---

# Summary

We introduce a Python open-source library for $\mathcal{X}$-armed bandit and online blackbox optimization named PyXAB. 
PyXAB contains the implementations of more than 10 $\mathcal{X}$-armed bandit algorithms, such as \texttt{Zooming}, 
\texttt{StoSOO}, \texttt{HCT}, and the most recent works such as \texttt{GPO}  and \texttt{VHCT}. PyXAB also provides 
the most commonly-used synthetic objectives to evaluate the performance of different algorithms and the various choices 
of the hierarchical partitions on the parameter space. The online documentation for PyXAB includes clear instructions 
for installation, straightforward examples, detailed feature descriptions, and a complete reference of the API. 
PyXAB is released under the MIT license in order to promote both academic and industrial usage. The library can be 
directly installed from PyPI with its source code available at \url{https://github.com/WilliamLwj/PyXAB}.

# Statement of need

Online blackbox optimization has become a heated research topic due to the recent popularity of machine learning models 
and thus the increasing demand for hyper-parameter tuning algorithms [@Li2018Hyperband; @shang2019general]. 
Other applications, such as neural architecture search, federated learning, and personal investment portfolio designs, 
also contribute to its prosperity nowadays [@li2021optimumstatistical; @Li2022Federated]. Different online blackbox 
optimization algorithms, e.g., Bayesian Optimization algorithms [@Shahriari2016Taking] and two-point evaluation methods
[@DuchiOptimal; @Shamir2015An] have been proposed. 




Apart from the aforementioned works, another very famous line of research is $\mathcal{X}$-armed bandit, also known as 
Lipschitz bandit, global optimization or bandit-based blackbox optimization [@kleinberg2008multi-armed; @bubeck2011X; @Grill2015Blackbox; @bartlett2019simple].
In this field, researchers split the parameter domain $\mathcal{X}$ into smaller and smaller sub-domains 
(commonly known as nodes) hierarchically, and treat each sub-domain to be an un-evaluated arm as in the multi-armed 
bandit problems [@bubeck2011X; @azar2014online].  However, such $\mathcal{X}$-armed bandit problems are much harder than
their multi-armed counterparts, since the number of sub-domains increases exponentially as the partition grows, and the 
hierarchical structure/Lipschitzness assumption implies internal correlations between the ``arms". Therefore, directly 
applying multi-armed bandit algorithms to such problems would infeasible and more complicated algorithms are developed 
[@Grill2015Blackbox; @shang2019general; @li2021optimumstatistical]. 

Despite the popularity of this area, most of the algorithms proposed by the researchers are either not open-sourced or 
implemented in different programming languages in disjoint packages. For example, \texttt{StoSOO}[@Valko13Stochastic] is
implemented in MATLAB and C\footnote{\url{https://team.inria.fr/sequel/software/}}, whereas \texttt{HOO} [@bubeck2011X] 
is implemented in Python\footnote{\url{https://github.com/ardaegeunlu/X-armed-Bandits}}. For most of the other algorithms,
no open-sourced implementations could be found on the internet. We believe the lack of such resources results from the 
following two main reasons. 

 - The algorithms are long and intrinsically hard to implement due to the heavy usage of hierarchical partitions,
node sampling, and the exploration-exploitation strategies that involve building, maintaining, and expanding complicated
tree structures. It is hence time-consuming to implement and test one single algorithm.
    
 - The problem settings for the algorithms could be slightly different.  Some 
algorithms such as \texttt{HOO} [@bubeck2011X] and \texttt{HCT} [@azar2014online] are designed for the setting where the
function evaluations can be noisy, while \texttt{SequOOL} [@bartlett2019simple] is proposed for the noiseless case. 
Some algorithms focus on cumulative-regret optimization whereas some only care about the last-point regret or the simple regret\footnote{ A more detailed discussion on simple
regret and cumulative regret can be found in [@bubeck2011X]}. Therefore, experimental comparisons often focus on a small
subset of algorithms, see e.g., [@azar2014online], [@bartlett2019simple]. The unavailability of a general package only 
deteriorates the situation. In Table \ref{tab: summary}, we provide the comparison among \texttt{HOO} [@bubeck2011X],  
\texttt{DOO} [@Munos2011Optimistic], \texttt{StoSOO} [@Valko13Stochastic], \texttt{HCT} [@azar2014online], \texttt{POO} [@Grill2015Blackbox]
\texttt{GPO} [@shang2019general], \texttt{SequOOL} [@bartlett2019simple],  \texttt{StroquOOL} [@bartlett2019simple],  \texttt{VROOM} [@ammar20derivative].
and \texttt{VHCT} [@li2021optimumstatistical].

\begin{table}
    \centering
    \caption{Selected examples of $\mathcal{X}$-armed bandit algorithms implemented in our library. \textit{Cumulative}: whether the algorithm focuses on optimizing cumulative regret or simple regret. \textit{Stochastic}: whether the algorithm deals with noisy rewards. \textit{Open-sourced?}:  the code availability before the development of PyXAB.}
    \begin{tabular}{l c c c}
        \hline
        {$\mathcal{X}$-Armed Bandit Algorithm} 
        & Cumulative & Stochastic & {Open-sourced?}   \\
        \hline
         \texttt{HOO} & yes & yes & yes (Python)  \\
        \texttt{DOO}   & no & no & no  \\
        \texttt{StoSOO}   & no & yes & yes (MATLAB, C)  \\
        \texttt{HCT}   & yes & yes & no   \\
        \texttt{POO}  & no & yes & yes (Python, R) \\
        \texttt{GPO}   & no & yes & no  \\
        \texttt{SequOOL}   & no & no &no \\
        \texttt{StroquOOL}    & no & yes &no \\
        \texttt{VROOM}   & no & no &no \\
        \texttt{VHCT}   & yes & yes &no\\
       \hline
    \end{tabular}
    \label{tab: summary}
\end{table}

To remove the barriers for future research in this area, we have developed PyXAB, a Python library of the existing 
popular $\mathcal{X}$-armed bandit algorithms. To the best of our knowledge, this is the first comprehensive library for
$\mathcal{X}$-armed bandit, with clear documentations and user-friendly API references. 


# Library Design and Usage
![An overview of the PyXAB library structure.\label{fig: overview}](Presentation1.pdf)


The API of PyXAB is designed to follow the $\mathcal{X}$-armed bandit learning paradigm and to allow the maximum 
freedom of usage. We provide an overview of the library in Figure \autoref{fig: overview}. 

\textbf{Algorithm}. All the algorithms inherit the abstract class \texttt{Algorithm}. Each algorithm will implement two 
kinds of actions via: (1) a \texttt{pull()} function that returns the chosen point to be evaluated by the objective; 
(2) a \texttt{receive\_reward()} function to collect the evaluation result and update the algorithm behavior. 

\textbf{Partition}. Given any parameter domain, the user is able to choose any partition of the domain as part of the
input of the optimization algorithm. All implemented partitions inherit the \texttt{Partition} class, which has useful 
base functions such as \texttt{deepen} and \texttt{get\_node\_list}. Each specific partition class needs to implement a 
unique \texttt{make\_children()} function that split one parent node into the children nodes and maintain the tree 
structure. For implementation convenience, the package also provides a few built-in choices such as 
\texttt{BinaryPartition} and \texttt{RandomBinaryPartition}.

\textbf{Node}. The base node class used in any partition is \texttt{P\_node}, which contains useful helper functions to 
store domain information and maintain the partition structure. However, we allow the algorithms to overwrite the node 
choices in any partition so that node-wise operations are allowed. For example, the \texttt{StoSOO} algorithm needs to 
compute and store the $b_{h,i}$-value for each node [@Valko13Stochastic]. The \texttt{StroquOOL} algorithm needs to 
record the number of times a node is opened [@bartlett2019simple]. Therefore, different node classes are implemented for these algorithms.

\textbf{Objective}. For all the objectives implemented in this package, they all inherit the \texttt{Objective} class 
and all have a function \texttt{f()} that returns the evaluation result of a given point. We also provide the commonly 
used synthetic objectives which are used to evaluate the performance of $\mathcal{X}$-armed bandit algorithms in research
papers, such as \texttt{Garland}, \texttt{DoubleSine}, and \texttt{Himmelblau}.




The usage of the PyXAB library is rather straightforward. Given the number of rounds, the objective function, 
and the parameter domain, the learner would choose the partition of the parameter space and the bandit algorithm. 
Then in each round, the learner obtains one point from the algorithm, evaluate it on the objective, and return the 
reward to the algorithm.
The following snippet of code provides an example of optimizing the Garland synthetic objective on the domain 
$[[0, 1]]$ by running the \texttt{HCT} algorithm with \texttt{BinaryPartition} for 1000 iterations. 
As can be observed, only about ten lines of code are needed for the optimization process apart from the import statements.


```python
from PyXAB.synthetic_obj.Garland import Garland
from PyXAB.algos.HCT import HCT

# Define the number of rounds, target, domain, and algorithm
T = 1000
target = Garland()
domain = [[0, 1]]
algo = HCT(domain=domain)

# Run the algorithm HCT
for t in range(1, T+1):
    point = algo.pull(t)
    reward = target.f(point)
    algo.receive_reward(t, reward)
```

# Code Quality and Documentations

In order to ensure high code quality, we follow the \texttt{PEP8} style and format all of our code using the 
\texttt{black} package\footnote{\url{https://github.com/psf/black}}. We use the \texttt{pytest} package to test our 
implementations with different corner cases. More than 99\% of our code is covered by the tests and Github workflows 
automatically generate a coverage report upon each push or pull request on the main branch\footnote{\url{https://github.com/WilliamLwj/PyXAB}}.

We provide detailed API documentations for each of the implemented classes and functions through numpy docstrings. 
The documentation is fully available online on ReadTheDocs\footnote{\url{https://pyxab.readthedocs.io/}}. 
On the same website, we also provide installation guides, algorithm introductions, both elementary and advanced examples 
of using our package, as well as detailed contributing instructions and new feature implementation examples to encourage
future contributions.

# References