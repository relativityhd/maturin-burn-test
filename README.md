# Maturin Burn Test

A test weather `burn` works together with `maturin` / `PyO3`, especially in different environments like Windows and Linux.

## Results

It works! :tada:

I wanted to check weather the following workflow works:

1. Install the library (which is a CLI) into an python environment on "any" machine. I checked the following machines:
    - Windows with dedicated GPU
    - Windows with integrated GPU
    - Linux (WSL) with dedicated GPU
    - Linux (WSL) with integrated GPU
2. Run the CLI with two numerical arguments and an optional `--gpu` flag
3. CLI creates `numpy` arrays based on arguments
4. Python calls `PyO3` Rust code and passes the numpy arrays
5. Rust converts `PyO3-numpy` arrays into `slice`, then into `burn` data objects and then into `burn` tensor objects
    - Depending on the `--gpu` flag, the tensors are either created with the `WGPU` backend or with the `NDArray` backend
7. Doing a calculation (simple `add`) with the tensors and store the result in a new tensor
8. Converting the new tensor all the way back to numpy and printing the contents from python

I know there is a lot of convertion happening, probably a lot more behind the scenes of `PyO3` and `burn`.
However, this workflow would allow the creation of fancy CLIs and Python libraries with a focus on scientific and heavy computing, while remaining easy to install and use.

## Run by yourself

I am using rye, but it should work with pip etc. as well.

```sh
rye add maturin-burn-test --git https://github.com/relativityhd/maturin-burn-test
```

After that you can use the CLI to test weather it works in your environment:

```sh
$ rye run maturin-burn-test calc 10 1 --gpu
GPU device: BestAvailable
[ 0.         1.2222223  2.4444447  3.6666665  4.8888893  6.1111107
  7.333333   8.555555   9.777779  11.       ]
```

Or import the `pyadd` function in your code:

```py
>>> from maturin_burn_test import pyadd
>>> import numpy as np
>>> 
>>> a = np.array([1, 2, 3], dtype=np.float32)
>>> b = np.array([4, 5, 6], dtype=np.float32) 
>>> use_gpu = True
>>> pyadd(use_gpu, a, b)
GPU device: BestAvailable
array([5., 7., 9.], dtype=float32)
```

## Dev-Setup

Prereq:

- [rust](https://www.rust-lang.org/)
- [rye](https://rye.astral.sh/)
- [maturin](https://www.maturin.rs/)

You can install maturin with rye:

```sh
rye install maturin
```

Then you can sync your environment:

```sh
rye sync
```

After each change to the rust code you need to recompile, so that python can import the lowlevel rust code:

```sh
maturin develop
```

## Big Thanks

How awesome how much is possible! A big thank you to all the contributors of the following libraries, which are used in this test:

- [Burn](https://burn.dev/)
- [Maturin](https://www.maturin.rs/)
- [PyO3](https://github.com/PyO3/pyo3)
- [NDArray](https://github.com/rust-ndarray/ndarray)
- [Typer](https://typer.tiangolo.com/)
- [Rye](https://rye.astral.sh/)
- [WGPU](https://wgpu.rs/)
