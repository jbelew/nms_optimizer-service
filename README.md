# No Man's Sky Starship Optimizer (Service Layer)

## CI/CD Status

![Tests](https://github.com/jbelew/nms_optimizer-service/actions/workflows/main.yml/badge.svg?branch=main)
![Deployment](https://img.shields.io/badge/Deployment-Heroku-blue?logo=heroku)

This repository contains the code for the `nms_optimizer-service`, a service designed to optimize the placement of technology modules within the inventory grid of No Man's Sky starships. It leverages various algorithms, including pattern matching and simulated annealing, to maximize adjacency bonuses and overall effectiveness of the installed technologies.

## Key Features

*   **Module Placement Optimization:**  Optimizes the placement of modules within a grid to maximize adjacency bonuses and supercharged slot utilization.
*   **Pattern Matching:** Utilizes pre-defined patterns (solves) for known optimal configurations and adapts them to the user's grid.
*   **Simulated Annealing:** Employs a simulated annealing algorithm to refine module placement and explore alternative configurations for improved scores.
*   **Technology Tree Generation:** Generates a technology tree for a given ship, providing a structured view of available technologies and their relationships.
*   **REST API:** Provides REST API endpoints for grid optimization and technology tree retrieval.

## Getting Started

### Prerequisites

*   Python 3.x
*   `pip` (Python package installer)
*   `virtualenv` (recommended for creating virtual environments)

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/nms_optimizer-service.git
    ```
2.  Navigate to the project directory:
    ```bash
    cd nms_optimizer-service
    ```
3.  **Create and activate a virtual environment (recommended):**
    ```bash
    source venv/bin/activate  # Activate the virtual environment (Linux/macOS)
    ```
4.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Service

1. **Ensure your virtual environment is activated.**
2. To start the service locally, run:

```bash
python app.py
