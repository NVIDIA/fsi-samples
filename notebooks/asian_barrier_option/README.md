
## Asian Barrier Options Pricing using GPU Acceleration

### Introduction

The European and American Options price can be estimated accurately by the efficient [Black–Scholes model](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model). Options like [Barrier Option](https://en.wikipedia.org/wiki/Barrier_option) and [Basket Option](https://en.wikipedia.org/wiki/Basket_option) have a complicated structure with no simple analytical solution. The Monte Carlo simulation is an effective way to price them. To get an accurate price with a small variance, a large number of simulation paths are needed which is computationally intensive. Luckily, each of the simulation paths are independent and we can take advantage of the multiple core GPU to accelerate the computation. Using GPU can speedup the computation by orders of magnitude due to the parallelization of the independent paths. But even that is still not fast enough. Recently, [Deep learning derivatives method](https://arxiv.org/pdf/1809.02233.pdf) was introduced to value derivatives and achieves speedup even higher than the former.  

In this tutorial, we are going to price the [Down-and-Out](https://www.investopedia.com/terms/d/daoo.asp) [Asian](https://www.investopedia.com/terms/a/asianoption.asp) [Barrier](https://www.investopedia.com/terms/b/barrieroption.asp) [Call Option](https://www.investopedia.com/terms/c/calloption.asp) :

    
### Barrier Option pricing

Asian Barrier Option is a mixture of [Asian Option](https://en.wikipedia.org/wiki/Asian_option) and [Barrier Option](https://en.wikipedia.org/wiki/Barrier_option). The price depends on the average underlying Asset Price `S`, the Strick Price `K` and the Barrier Price `B`. There are 4 types of Barrier Options:-
   * [Up-and-out](https://www.investopedia.com/terms/u/up-and-outoption.asp): spot price starts below the barrier level and has to move up for the option to be knocked out.
   * [Down-and-out](https://www.investopedia.com/terms/d/daoo.asp): spot price starts above the barrier level and has to move down for the option to be knocked out.
   * [Up-and-in](https://www.investopedia.com/terms/u/up-and-inoption.asp): spot price starts below the barrier level and has to move up for the option to become activated.
   * [Down-and-in](https://www.investopedia.com/terms/d/daio.asp): spot price starts above the barrier level and has to move down for the option to become activated.

Without loss of generality, in this tutorial we will use the [Down-and-Out Call Discretized Asian Barrier Option](https://ieeexplore.ieee.org/document/6327776/metrics#metrics) as an example. The option will be void if the average price of the underlying asset goes below the barrier. The asset Spot Price `S` is usually modeled as [Geometric Brownian motion](https://en.wikipedia.org/wiki/Geometric_Brownian_motion), which has 3 free parameters:- [Spot Price](https://www.investopedia.com/terms/s/spotprice.asp), [Percent Volatility](https://www.investopedia.com/terms/v/volatility.asp) and the [Percent Drift](https://en.wikipedia.org/wiki/Stochastic_drift). The price of the option will be the expected profit at the maturity discount to the current value.

### Preliminary 

You need to build a docker image to run the examples. 

```bash
cd docker
build -f Dockerfile -t option .
# launch your nvidia docker container and expose the port for Jupyterlab
```

### Outline 

This tutorial is organized as following notebooks

1. [Use Python GPU libraries to accelerate the Monte Carlo pricing on the GPU](./mc_pricing.ipynb)
2. [Use the Monte Carlo pricing dynamic dataset to train an Option Pricing Neural Network Model](./deep_learning_option_1.ipynb)
3. [Use the Monte Carlo pricing staic dataset to train an Option Pricing Neural Network Model and do inference](./deep_learning_option_2.ipynb)
4. [Train an Asian Barrier Option Pricing Neural Network Model with NeMo](./deep_learning_nemo.ipynb)
5. [Accelerate the Option Pricing Neural Network Model inference with TensorRT](./tensorrt.ipynb)
