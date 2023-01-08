# GP code, details, links shared by Seniors
The notebooks in this directory contain toy codes and simple pollution codes for usage of gpytorch and gptorch. The toy example has some wind/temp data that the code generates. The code may not be ordered properly as seniors couldn't make gp run anyway with Crypten. The Crypten versions of the toy examples had the functions in libraries modified to take care of MPC.

These two original notebooks were shared by Riju Ma'am.
1. [Toyexample_Gpytorch.ipynb](https://colab.research.google.com/drive/1iCI3NS5eHg5O45ZTXzx9X9Ps2pVYXfRx?usp=sharing)
2. [Pollution_Gpytorch.ipynb](https://colab.research.google.com/drive/11aNay3faLruCThIEFYFzrmhyuj-Hl46w?usp=sharing)

The sections below contain resources and links that were shared by seniors.

## Libraries used for GP
1. [GPyTorch](https://gpytorch.ai/): This was the library used by seniors for GP.
## Libraries identified/to be explored
Contains links for libraries which were identified but not used by seniors.
1. [gptorch](https://github.com/cics-nd/gptorch/tree/master/gptorch): based on pytorch
2. [GPflow](https://github.com/GPflow/GPflow)
3. [GPy](https://github.com/SheffieldML/GPy)

Libraries `GPflow` and `GPy` were not based on torch hence it would be difficult to make them work with CrypTen.

## Resources
1. [VGP](https://towardsdatascience.com/variational-gaussian-process-what-to-do-when-things-are-not-gaussian-41197039f3d4): Nice resource to learn more about GP.
2. [ML Tutorial GP](https://www.youtube.com/watch?v=92-98SYOdlY&ab_channel=MarcDeisenroth): Gaussian Process explained!

## Errors to solve
1. covar_x<class 'gpytorch.lazy.lazy_evaluated_kernel_tensor.LazyEvaluatedKernelTensor'> not supported by cryptensor
2. not isinstance(model, GP) -> raises exception in gpytorch\gpytorch\mlls\marginal_log_likelihood.py
3. 'MPCTensor' object has no attribute 'ndimension'
4. CrypTen does not support torch function <method 'sub' of 'torch._C._TensorBase' objects>

## Updated gpytorch library for Crypten
Install the updated gpytorch library for running the GP code in crypten. To intsall this version do- pip install -e /path_to_gpytorch
