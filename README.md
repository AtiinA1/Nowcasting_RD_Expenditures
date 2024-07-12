# Nowcasting R&D Expenditures

## Overview

This repository contains the implementation of a neural network-based nowcasting model to predict and interpolate research and development (R&D) expenditure data using mixed-frequency, high-dimensional data sources such as internet search volume data. The project addresses the 'ragged-edge' problem in macroeconomic data acquisition, providing timely and high-frequency estimates of R&D investments, which are traditionally collected through infrequent and delayed surveys.

## Framework

The framework comprises two main steps:
1. **Nowcasting Model (Step A):** A supervised learning model leveraging both low-frequency and high-frequency data to predict observed low-frequency R&D expenditures.
2. **Temporal Interpolation (Step B):** An unsupervised learning task that uses elasticities derived from Step A to interpolate unobserved high-frequency figures, producing monthly estimates of R&D expenditures.

## Key Components

- **Data Collection:** Utilizes a range of predictors, predominantly internet search volume data, to enhance prediction accuracy.
- **Neural Network Model:** Implements a neural network-based approach for nowcasting yearly R&D expenditures.
- **Interpolation Method:** Allocates yearly R&D investments into monthly figures based on model-derived elasticities.
- **Comparison and Validation:** Compares the results with classical regression-based methods and validates them using monthly R&D employment data.

## Repository Structure

- `data/`: Contains the datasets used for training and validation.
- `gt_code/`: Includes scripts for fetching and processing Google Trends data.
- `nn_mlp_nowcasting_model/`: Contains the implementation of the neural network-based nowcasting model.
- `temporal_disaggregation/`: Implements the temporal interpolation techniques for high-frequency estimation.

## Installation and Usage

1. **Clone the repository:**
   ```sh
   git clone https://github.com/AtiinA1/Nowcasting_RD_Expenditures.git
   cd Nowcasting_RD_Expenditures

## Contact
   
For any questions or issues, please do not hesitate to contact [Atin Aboutorabi](https://people.epfl.ch/atin.aboutorabi?lang=en) :)
