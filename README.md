# 🧊 Interactive Gravitational Force Simulation Between Two Cubes

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.37%2B-red?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

An interactive web application built with Streamlit that calculates and visualizes the gravitational force between two touching unit cubes. This project provides an educational walkthrough of solving a classic numerical analysis problem (Trefethen's Problem #5) using the elegant Prideaux method combined with Gaussian Quadrature.

---

## ## Overview

Calculating the gravitational attraction between two cubes that are in direct contact is a surprisingly difficult problem. A direct integration approach fails due to a singularity where the cubes touch (the distance between points becomes zero).

This application demonstrates the solution proposed by J. Prideaux (2002), which cleverly circumvents the singularity by:

1.  **Decomposing** each cube into 8 smaller sub-cubes.
2.  **Classifying** the 64 resulting sub-cube pairs into types: Face, Edge, Vertex, and Separated.
3.  **Using recursion and scaling laws** to solve for the forces between touching pairs based on the forces of the easily calculable "Separated" pairs.
4.  **Employing Gaussian Quadrature**, accelerated with Numba, for the high-dimensional numerical integration required for the separated pairs.

---

## ## Key Features

- **Interactive Simulation:** Adjust the order of the Gaussian Quadrature in real-time to see its effect on accuracy and computation time.
- **Step-by-Step Explanation:** The app is structured with tabs that guide the user from the initial problem statement to the final result.
- **Rich Visualizations:** 3D plots of the cube configurations, diagrams of the Prideaux decomposition, and charts illustrating the numerical methods.
- **High-Precision Result:** Accurately calculates the final force to approximately 10 decimal places, matching the known literature value.

---

## ## Getting Started

Follow these instructions to get a local copy up and running.

### ### Prerequisites

- Python 3.9 or higher
- Git

### ### Installation

1.  **Clone the repository:**

    ```sh
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
    cd YOUR_REPOSITORY_NAME
    ```

2.  **Create and activate a virtual environment:**

    - **Windows:**
      ```sh
      python -m venv .venv
      .\.venv\Scripts\Activate
      ```
    - **macOS / Linux:**
      ```sh
      python -m venv .venv
      source .venv/bin/activate
      ```

3.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

---

## ## Usage

To run the Streamlit application, execute the following command in your terminal:

```sh
streamlit run app.py
```
