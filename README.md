#  Evaluating the Worst Case Approximation Error for Convex Relaxations to Multilayer Perceptrons
## Abstract

> In this work we present a worst case analysis, for convex relaxations to multilayer perceptrons. Formal reasoning on Multilayer Perceptrons (MLPs) is computationally intractable, due to the integrality introduced by the ReLU activation. Recent works employ convex relaxations, thus avoiding integer programming. This method achieves better performance, but also introduces approximation errors. Our work examines the worst case approximation error. We begin by reviewing the lattice-structured space of feasible convex relaxations. We focus on the tight convex approximation, where every neuron is linearly relaxed. The tight relaxation represents the feasible, worst case scenario. The tight divergence measures the ℓ∞-distance between the tight relaxation’s and original MLP’s outputs for a particular input. Furthermore, we discuss the worst case error, which is the maximum value of the tight divergence. We present theoretical lower and upper upper bounds, predicting an exponential escalation of the worst case error, w.r.t. the number of layers. Our experimental results support this claim, evidencing an exponential growth, also for the average error. Furthermore, we perform tests on a real life MNIST network. There we observe linear escalation w.r.t. the input radius. Finally, we review how the divergence is reflected in classification problems, observing that the mis-classification probability increases linearly w.r.t. the input radius. Our findings suggest that convex relaxation produces accurate results only for small networks with short input radius.

*Under Review for the 23rd International Conference on Principles of Knowledge Representation and Reasoning.*

## Installation

Clone the github repository, using `git clone <repo-url>`. Then `cd` to the `./worst-case-convex-approx-error` directory. Finally, create a dedicated [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html) environment using the following command:

```bash
conda env create -n convex-env -f convex-environment.yml
```

## Execution

Run the notebooks in `./experiments` directory.

| File                      | Description                                                  |
| ------------------------- | ------------------------------------------------------------ |
| `simple.ipynb`            | Testing the escalation of the convex approximation error, w.r.t. the network's depth. On random MLPs, for input domain radius $\rho = 1$. |
| `simple-rad-0.ipynb`      | Testing the escalation of the convex approximation error, w.r.t. the network's depth. On random MLPs, for input domain radius $\rho = 0$. |
| `simple-rad-0.025.ipynb`  | Testing the escalation of the convex approximation error, w.r.t. the network's depth. On random MLPs, for input domain radius $\rho = 0.025$. |
| `mnist.ipynb`             | Testing the escalation of the convex approximation error, and the mis-classification probability, w.r.t. the input domain radius. On a MLP trained on MNIST. |
| `fashion-mnist.ipynb`     | Testing the escalation of the convex approximation error, and the mis-classification probability, w.r.t. the input domain radius. On a MLP trained on Fashion MNIST. |
| `single-perceptron.ipynb` | A running example exhibiting the behavior of the convex approximation on a single perceptron. |

