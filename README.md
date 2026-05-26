# HI-PINN: Hybrid Iterative Physics-Informed Neural Networks for Inverse Source Problems

## Overview

This repository contains the complete code and data for the paper:

**"Hybrid Iterative Physics-Informed Neural Networks for Ill-Posed Inverse Source Problems"**  
Submitted to *Mathematical Methods in the Applied Sciences*  
Special Issue: *Inverse Problems, Numerical Methods and Artificial Intelligence*

## Abstract

This code implements a hybrid framework that combines classical Landweber iteration with physics-informed neural networks (PINNs) to solve ill-posed inverse source problems for the 1D Poisson equation. The method includes ensemble-based uncertainty quantification and achieves 79.4% error reduction compared to classical methods.

## Key Results

| Method | Relative L2 Error |
|--------|------------------|
| Classical Landweber | 99.51% |
| HI-PINN (Ours) | 20.52% |

**Improvement: 79.4% reduction**

## Requirements

- Python 3.8+
- Google Colab (recommended) or local GPU
- Required packages listed in `requirements.txt`

## Quick Start (Google Colab - 5 minutes)

1. Open [Google Colab](https://colab.research.google.com)
2. Click **File → Upload Notebook**
3. Upload `hi_pinn_inverse_problem.ipynb`
4. Click **Runtime → Run all**
5. Results will appear in the final cell

## Local Installation

```bash
git clone https://github.com/YOUR_USERNAME/hi-pinn-inverse-problems.git
cd hi-pinn-inverse-problems
pip install -r requirements.txt
python hi_pinn_inverse_problem.py

hi-pinn-inverse-problems/
├── README.md                          # This file
├── hi_pinn_inverse_problem.ipynb      # Main Colab notebook
├── hi_pinn_inverse_problem.py         # Python script version
├── requirements.txt                   # Dependencies
├── final_results.json                 # Numerical results
├── manuscript_figures.png             # Publication figure
└── LICENSE                            # MIT License

Output Files
After running the code, you will get:

File	Description
manuscript_figures.png	Figure with 6 subplots (true source, reconstruction, loss, uncertainty, error bars, improvement)
final_results.json	Structured numerical results
Figure Description
The generated figure contains 6 subplots:

Top left: Ground truth source function (Gaussian peak at x=0.5)

Top middle: Reconstruction comparison (True vs HI-PINN vs Landweber)

Top right: Training loss convergence (log scale, 2000 epochs)

Bottom left: Uncertainty quantification (95% confidence interval)

Bottom middle: Error comparison bar chart

Bottom right: Error reduction pie chart (79.4%)

Numerical Results (from final_results.json)
json
{
  "l2_error_pinn_percent": 20.52,
  "l2_error_landweber_percent": 99.51,
  "improvement_absolute": 78.99,
  "improvement_relative": 79.4,
  "final_loss": 2.095e-05,
  "mean_uncertainty": 0.019038,
  "max_uncertainty": 0.052504,
  "correlation": 0.149,
  "n_sensors": 15,
  "noise_level_percent": 3.0,
  "nx": 100,
  "iterations": 2000
}
Citation
If you use this code, please cite:

text
[Murtaza Aqeel]. (2026). HI-PINN: Hybrid Iterative Physics-Informed Neural Networks 
for Inverse Source Problems. GitHub. 
https://github.com/YOUR_USERNAME/hi-pinn-inverse-problems
License
MIT License - Free for academic and commercial use.

Contact
[Murtaza Aqeel] - [murtazamandsore7@gmail.com]

Acknowledgments
The author thanks the guest editors of the special issue—Prof. Vladislav V. Kravchenko, Prof. Arutyun Avetisyan, Prof. Evgeny Burnaev, and Prof. Alexey Karapetyants.

text

---

## File 5: `final_results.json`

```json
{
  "l2_error_pinn_percent": 20.52,
  "l2_error_landweber_percent": 99.51,
  "improvement_absolute": 78.99,
  "improvement_relative": 79.4,
  "final_loss": 2.095e-05,
  "mean_uncertainty": 0.019038,
  "max_uncertainty": 0.052504,
  "correlation": 0.149,
  "n_sensors": 15,
  "noise_level_percent": 3.0,
  "nx": 100,
  "iterations": 2000
}
