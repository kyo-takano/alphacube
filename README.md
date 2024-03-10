# AlphaCube

AlphaCube is a powerful & flexible Rubik's Cube solver that extends [EfficientCube](https://github.com/kyo-takano/efficientcube).

## App

We offer an interactive demo application for you to try out AlphaCube:
[alphacube.dev](https://alphacube.dev)

## Installation

Open a terminal and execute the following command:

```sh
pip install alphacube
```

## Usage

### Basic

Using AlphaCube in Python is as simple as this:

```python
import alphacube

# Load a trained DNN
alphacube.load("small") # the default model

# Solve the cube using a given scramble
result = alphacube.solve(
    scramble="D U F2 L2 U' B2 F2 D L2 U R' F' D R' F' U L D' F' D R2",
    beam_width=1024,
)
print(result)
```

> **Output**
> ```python
> {
>     'solutions': [
>         "D L D2 R' U2 D B' D' U2 B U2 B' U' B2 D B2 D' B2 F2 U2 F2"
>     ],
>     'num_nodes': 19744,
>     'time':1.4068585219999659
> }
> ```

AlphaCube selects the smallest model by default and takes about a few seconds to find a moderately good solution.

### Better Solutions

If you want even shorter solutions, simply increase the `beam_width` parameter:

```python
alphacube.load() # model_id="small"
result = alphacube.solve(
    scramble="D U F2 L2 U' B2 F2 D L2 U R' F' D R' F' U L D' F' D R2",
    beam_width=65536,
)
print(result)
```

> **Output**
> ```python
> {
>     'solutions': [
>         "D' R' D2 F' L2 F' U B F D L D' L B D2 R2 F2 R2 F'",
>         "D2 L2 R' D' B D2 B' D B2 R2 U2 L' U L' D' U2 R' F2 R'"
>     ],
>     'num_nodes': 968984,
>     'time': 45.690575091997744
> }
> ```

You generally get more and better solutions, at the sacrifice of an increased wall-clock time and computational cost.

### Applying Ergonomic Bias

```python
ergonomic_bias = {
    "U": 0.9,   "U'": 0.9,  "U2": 0.8,
    "R": 0.8,   "R'": 0.8,  "R2": 0.75,
    "L": 0.55,  "L'": 0.4,  "L2": 0.3,
    "F": 0.7,   "F'": 0.6,  "F2": 0.6,
    "D": 0.3,   "D'": 0.3,  "D2": 0.2,
    "B": 0.05,  "B'": 0.05, "B2": 0.01,
    "u": 0.45,  "u'": 0.45, "u2": 0.4,
    "r": 0.3,   "r'": 0.3,  "r2": 0.25,
    "l": 0.2,   "l'": 0.2,  "l2": 0.15,
    "f": 0.35,  "f'": 0.3,  "f2": 0.25,
    "d": 0.15,  "d'": 0.15, "d2": 0.1,
    "b": 0.03,  "b'": 0.03, "b2": 0.01
} # Each value represents the desirability score

result = alphacube.solve(
    scramble="D U F2 L2 U' B2 F2 D L2 U R' F' D R' F' U L D' F' D R2",
    beam_width=65536,
    ergonomic_bias=ergonomic_bias
)
print(result)
```

> **Output**
> ```python
> {
>     'solutions': [
>         "u' U' f' R2 U2 R' L' F' R D2 f2 R2 U2 R U L' U R L",
>         "u' U' f' R2 U2 R' L' F' R D2 f2 R2 U2 R d F' U f F",
>         "u' U' f' R2 U2 R' L' F' R u2 F2 R2 D2 R u f' l u U"
>     ],
>     'num_nodes': 1078054,
>     'time': 56.13087955299852
> }
> ```

### GPU Acceleration

If you have a GPU, we highly recommend you select the largest model for the best possible performance:

```python
alphacube.load("large")
result = alphacube.solve(
    scramble="D U F2 L2 U' B2 F2 D L2 U R' F' D R' F' U L D' F' D R2",
    beam_width=65536,
)
print(result)
```

> **Output**
> ```python
> {
>     'solutions': ["D F L' F' U2 B2 U F' L R2 B2 U D' F2 U2 R D'"],
>     'num_nodes': 903448,
>     'time': 20.46845487099995
> }
> ```

You will get even better solutions in an order of magnitude smaller amount of time.

On the contrary, using a model larger than `"small"` (i.e., `"base"` or `"large"`) *on CPU* will likely result in inferior temporal performance.

### Verbose Mode

Additionally, you may call `alphacube.set_verbose()` to keep track of the progress. It will display the current depth of search on your terminal.

Please refer to our [documentation](https://alphacube.dev/docs) for more, especially ["Getting Started"](https://alphacube.dev/docs/getting-started/index.html)

## How It Works

At the heart of AlphaCube lies a methodology rooted in the paper titled ["Self-Supervision is All You Need for Solving Rubik's Cube" (TMLR'23)](https://openreview.net/forum?id=bnBeNFB27b), the official code of which is also available as a [GitHub repository](https://github.com/kyo-takano/efficientcube).
This project releases the three largest solvers trained in Half-Turn Metric (HTM) as described in **Section 7. Scaling Law**.

In EfficientCube, a Deep Neural Network (DNN) is trained to predict the **probability distribution of the next moves** that would bring a given state one step closer to the goal.
These predicted moves are then applied sequentially to solve a given scramble.

This project releases the three largest, compute-optimally trained solvers as described in **Section 7. Scaling Law**.

You may also read ["How It Works"](https://alphacube.dev/docs/how-it-works/index.html) in the documentation for illustrated descriptions.

## Contributing

You are more than welcome to collaborate on AlphaCube.
If you're interested, please read our [CONTRIBUTING](https://github.com/kyo-takano/alphacube/blob/main/CONTRIBUTING.md) guide to get started.

## License

AlphaCube is open-sourced under the [MIT license](https://github.com/kyo-takano/alphacube/blob/main/LICENSE), granting you the freedom to use and modify it.
