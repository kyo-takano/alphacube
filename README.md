# AlphaCube

AlphaCube is a powerful & flexible Rubik's Cube solver that extends [EfficientCube](https://github.com/kyo-takano/efficientcube). It uses a Deep Neural Network (DNN) to find optimal/near-optimal solutions for a given scrambled state.

> [!NOTE]
> **🎮 Try the interactive demo: [alphacube.dev](https://alphacube.dev)**

## Use Cases

- Solve any scrambled Rubik's Cube configuration with ease
- Find efficient algorithms/solutions, optimizing for either computation speed or ergonomics of the move sequence
- Incorporate into Rubik's Cube apps and tools to provide solving capabilities
- Analyze and study the statistical properties and solution space of the Rubik's Cube puzzle
- Illustrate AI/ML concepts to students. Topics include:
  - discrete diffusion model
  - self-supervised learning
  - combinatorial search with probabilities

---

## Table of Contents

- [AlphaCube](#alphacube)
  - [Use Cases](#use-cases)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Basic](#basic)
    - [Better Solutions](#better-solutions)
    - [Applying Ergonomic Bias](#applying-ergonomic-bias)
    - [GPU Acceleration](#gpu-acceleration)
  - [How It Works](#how-it-works)
  - [Contributing](#contributing)
  - [License](#license)

## Installation

Open a terminal and execute the following command:

```sh
pip install -U alphacube
```

## Usage

### Basic

```python
import alphacube

# Load a trained DNN (default is "small" model)
alphacube.load()

# Solve the cube using a given scramble sequence
result = alphacube.solve(
    scramble="D U F2 L2 U' B2 F2 D L2 U R' F' D R' F' U L D' F' D R2",
    beam_width=1024, # Number of candidate solutions to consider at each depth of search
)
print(result)
```

> **Output**
>
> ```python
> {
>     'solutions': [
>         "D L D2 R' U2 D B' D' U2 B U2 B' U' B2 D B2 D' B2 F2 U2 F2"
>     ],
>     'num_nodes': 19744, # Total search nodes explored
>     'time': 1.4068585219999659 # Wall-clock time in seconds
> }
> ```

### Better Solutions

Increasing `beam_width` explores more candidate solutions, producing shorter (better) solve sequences at the cost of increased computation:

```python
result = alphacube.solve(
    scramble="D U F2 L2 U' B2 F2 D L2 U R' F' D R' F' U L D' F' D R2",
    beam_width=65536,
)
print(result)
```

> **Output**
>
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

`beam_width` values between 1024-65536 typically offer a good trade-off between solution quality and speed. Tune according to your needs.

### Applying Ergonomic Bias

The `ergonomic_bias` parameter lets you specify the desirability of each move type, influencing the solver to favor certain moves over others:

```python
# Desirability scale: 0 (lowest) to 1 (highest)
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
}

result = alphacube.solve(
    scramble="D U F2 L2 U' B2 F2 D L2 U R' F' D R' F' U L D' F' D R2",
    beam_width=65536,
    ergonomic_bias=ergonomic_bias
)
print(result)
```

> **Output**
>
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

For maximum performance, use the `"large"` model on a CUDA-enabled GPU (requires [PyTorch](https://pytorch.org/get-started/locally/)):

```python
alphacube.load("large")
result = alphacube.solve(
    scramble="D U F2 L2 U' B2 F2 D L2 U R' F' D R' F' U L D' F' D R2",
    beam_width=65536,
)
print(result)
```

> **Output**
>
> ```python
> {
>     'solutions': ["D F L' F' U2 B2 U F' L R2 B2 U D' F2 U2 R D'"],
>     'num_nodes': 903448,
>     'time': 20.46845487099995
> }
> ```

Using a GPU provides an order of magnitude speedup over CPU especially for larger models.

> [!IMPORTANT] When running AlphaCube _on a CPU_, it's generally recommended to stick with the `"small"` model, as the larger `"base"` and `"large"` models would take considerably more time to find solutions.

Please refer to our [documentation](https://alphacube.dev/docs) for more, especially ["Getting Started"](https://alphacube.dev/docs/getting-started/index.html)

## How It Works

At the heart of AlphaCube lies a deep learning method described in ["Self-Supervision is All You Need for Solving Rubik's Cube" (TMLR'23)](https://openreview.net/forum?id=bnBeNFB27b), the official code of which is also available as [EfficientCube](https://github.com/kyo-takano/efficientcube).

The 3 provided models (`"small"`, `"base"`, and `"large"`) are **_compute-optimally trained_** in the Half-Turn Metric, This means the model sizes are scaled in tandem with the amount of training data to maximize prediction accuracy for a given computational budget. See Section 7 of the above-mentioned paper for details.

> [!NOTE]
> **📖 Read more: ["How It Works"](https://alphacube.dev/docs/how-it-works/index.html)**

## Contributing

You are more than welcome to collaborate on AlphaCube. Please read our [Contributing Guide](https://github.com/kyo-takano/alphacube/blob/main/CONTRIBUTING.md) to get started.

## License

AlphaCube is open source under the [MIT License](LICENSE).
