# No Man's Sky Neural Technology Optimizer (Service Layer)

## CI/CD Status

![Tests](https://github.com/jbelew/nms_optimizer-service/actions/workflows/main.yml/badge.svg?branch=main) ![Deployment](https://img.shields.io/badge/Deployment-Heroku-blue?logo=heroku)

This repository contains the code for the **`nms_optimizer-service`**, a backend service designed to optimize the placement of modules within the technology grid of No Man's Sky starships and multi-tools. It leverages multiple optimization strategies, including pattern matching with edge detection, machine learning (AI), and simulated annealing, to maximize adjacency bonuses and the overall effectiveness of installed technologies.

This service powers the UI available at [https://nms-optimizer.app](https://nms-optimizer.app).

## Key Features

- **Module Placement Optimization:** Optimizes the placement of modules within a grid to maximize adjacency bonuses and supercharged slot utilization.
- **Pattern Matching:** Utilizes pre-defined patterns (solves) for known optimal configurations and adapts them to the user's grid.
- **TensorFlow Models:** Attempts to solve placement using a collection of TensorFlow models.
- **Simulated Annealing:** Employs a simulated annealing algorithm to polish and refine module placement and explore alternative configurations for improved scores.
- **Technology Tree Generation:** Generates a technology tree for a given ship, providing a structured view of available technologies and their relationships.
- **REST API:** Provides REST API endpoints for grid optimization and technology tree retrieval.
- **Google Analytics Integration:** Provides an endpoint (`/analytics/popular_data`) to retrieve Google Analytics 4 (GA4) data for popular ship types and technologies.

## Documentation

Detailed documentation for this project can be found in the `/docs` directory.

## Getting Started

### Prerequisites

- Python 3.14+
- `pip` (Python package installer)
- `virtualenv` (recommended for creating virtual environments)

### Service Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/jbelew/nms_optimizer-service.git
    ```
2.  Navigate to the project directory:
    ```bash
    cd nms_optimizer-service
    ```
3.  **Create and activate a virtual environment (recommended):**
    ```bash
    virtualenv venv  # Create a virtual environment
    source venv/bin/activate  # Activate the virtual environment (Linux/macOS)
    ```
4.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Service

#### Development
1. **Ensure your virtual environment is activated.**
2. To start the service locally, run:

```bash
python3 -m src.app
```
This will start the Flask development server.

#### Production
The service is designed to be run with Gunicorn. The `Procfile` included in the repository is set up for this purpose:
```
gunicorn --preload --timeout 120 src.app:app --keep-alive 60 --worker-class gevent
```

### Training Installation
The model generation and training portion of the code requires significantly more imports and libraries and has been seperated from the main service venv to prevent issues with cloud deployments. To run any of the training code --

1.  Navigate to the project directory:
    ```bash
    cd nms_optimizer-service/scripts/training
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    virtualenv venv  # Create a virtual environment
    source venv/bin/activate  # Activate the virtual environment (Linux/macOS)
    ```
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
There are a number of shell scripts in the training directory that I use to generate data for various platforms and technology types. Be sure and set them to executable (chmod 755) to run them.

#### Training data generation

```
python -m scripts.training.generate_data --ship standard-mt --category \"Weaponry\" --tech bolt-caster"
````

#### Best layout finder
Iterates through the generated sample data to find the best layout for various different supercharged counts

```
python -m scripts.training.find_best_layout --ship standard --tech infra 3
````

#### Solve map generation

```
python -m scripts.debugging_utils.solve_map_generator --ship corvette --tech infra
```

## Data Definitions

The `src/data_definitions` directory contains key data used by the application:

* **`recommended_builds.py`**: Contains JSON configuration data for recommended builds for applicable platforms.
* **`solves/`**: Contains pre-computed solutions (solves) for various technology and ship combinations. These are used by the pattern matching algorithm.
* **`modules_data/`**: Contains JSON files with the definitions of all the technology modules for each platform.
