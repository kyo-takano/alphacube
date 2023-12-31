# Contributing to AlphaCube

Thank you for considering contributing to AlphaCube, an open-source software designed for solving the Rubik's Cube.
Your contributions are invaluable to the development of this project.

## Prerequisites

AlphaCube's foundation lies in EfficientCube, the state-of-the-art deep learning method introduced in the paper ["Self-Supervision is All You Need for Solving Rubik's Cube" (TMLR'23)](https://openreview.net/forum?id=bnBeNFB27b).

While you do not need to delve into deep learning theory or scaling laws, you should have a clear understanding of the core idea of the proposed method ── especially how the candidate paths are evaluated probabilistically. You may be able to understand it from the source code ([`alphacube/search.py`](https://github.com/kyo-takano/alphacube/blob/main/alphacube/search.py)) as well, but we recommend that you read the original paper.

On top of that, we encourage you to also explore the latter part of [**How It Works**](https://alphacube.dev/how-it-works/index.html). 
It sheds light on how AlphaCube generates ergonomic/speed-optimal solutions based on the EfficientCube foundation. 
Although we have not written any paper on this technique, it should not be so challenging once you know the foundation.

## Roadmap

There are multiple avenues for contributing to AlphaCube's enhancement. Here are some *non-exclusive* possibilities:

- **Speed Optimization:**
  Explore ways to enhance the current implementation for improved performance.

- **Rewriting in Other Languages:**
  Consider reimplementing AlphaCube in languages like C++ or JavaScript for enhanced speed and convenience.

- **Training a New Model:**
  Although the existing three models are sufficiently trained above the compute-optimal frontier (meaning slightly overtrained for their scales), **they are trained to solve exclusively in HTM**.
  Therefore, if we want a solver that accommodates slice moves like `M`, `S`, and `E` (which we cannot algorithmically simulate from flat moves),
  we need a set of new models specifically trained in this combinatorial space (Slice-Turn Metric).

- **Data Collection with Smart Cubes:**
  Contribute by collecting data on execution times per move, potentially conditional on previous moves. This data can significantly improve the quality of ergonomic solutions generated by AlphaCube.

- **Utilities:**
  Add utility functions, such as result recording or estimating optimal beams within a given time budget.

- **Testing and Bug Fixes:**
  Ensure correctness by incorporating test code and addressing straightforward bugs.

- **Search Algorithm Improvements:**
  Utilize theory-based adjustments to enhance the current beam search algorithm for better performance.

- **Input Validation:**
  Strengthen input validation mechanisms for increased robustness.

- **Code Cleanup:**
  Refactor and organize the codebase for improved readability and maintainability.

## Contributor's Guide

The AlphaCube project welcomes contributors of all skill levels. To get started, please follow these steps:

1. **Familiarize Yourself:**
   Get a grasp of the core methodology by reading the paper, documentation, and/or taking a look at the source code to see how the concepts are implemented.

2. **Implementation:**
   If you're enhancing AlphaCube's existing features or fixing bugs, refer to the source code for guidance. For new features, make sure to align your work with the overarching methodology.

3. **Submit a Pull Request:**
   Once you're satisfied with your changes, submit a pull request from your branch to the main AlphaCube repository. Provide a detailed description of your work, including its purpose and how it fits into the project.

Every contribution, regardless of its size, contributes to the advancement of AlphaCube. We're grateful for your dedication to improving the project. Thank you for being a part of it.
