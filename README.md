# HI-PINN: Hybrid Iterative Physics-Informed Neural Networks for Inverse Source Problems

## Overview

This repository contains the complete code and data for the paper:

**"Hybrid Iterative Physics-Informed Neural Networks for Ill-Posed Inverse Source Problems"**  
Submitted to *Mathematical Methods in the Applied Sciences*  
Special Issue: *Inverse Problems, Numerical Methods and Artificial Intelligence*

## Abstract

This code implements a hybrid framework that combines classical Landweber iteration with physics-informed neural networks (PINNs) to solve ill-posed inverse source problems for the 1D Poisson equation. The method includes ensemble-based uncertainty quantification and achieves 79.4% error reduction compared to classical methods.

## Requirements

- Python 3.8+
- Google Colab (recommended) or local GPU
- Required packages: torch, matplotlib, numpy, scipy

## Quick Start (Google Colab)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `hi_pinn_inverse_problem.ipynb`
3. Click Runtime → Run all
4. Results will be displayed in the final cell

## Local Installation

```bash
pip install torch matplotlib numpy scipy
python hi_pinn_inverse_problem.py
