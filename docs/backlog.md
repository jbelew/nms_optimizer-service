# Backlog

This document serves as a living backlog for all tasks and implementation plans related to the NMS Optimizer Service.

## Future Features

- **Support for more ship and multi-tool types**: The service should be updated to support new ship and multi-tool types as they are added to the game.
- **Support for more technology modules**: The service should be updated to support new technology modules as they are added to the game.
- **More detailed statistics**: Provide more detailed statistics on the optimization results, such as the contribution of each module to the total bonus.

## Improvements

- **Performance Optimization**: The optimization algorithm can be further optimized to improve its speed and accuracy.
- **ML Model Improvement**: The machine learning model can be retrained with a larger dataset to improve its prediction accuracy.
- **Code Refactoring**: The codebase can be refactored to improve its readability and maintainability.
- **Window Size Refactoring**: Move window size logic out of `helpers.py` into the JSON definitions (In Progress).
- **Dependabot Workflow**: Add `.github/dependabot.yml` to automate updates for Python (pip), Node (npm), Rust (cargo), and GitHub Actions.
- **Pull Request Workflows**: Update GitHub Actions workflow to run the test suite on pull requests while restricting release and deploy stages to pushes on main.
- **Grouped Dependabot Updates**: Configure Dependabot to group minor and patch updates to minimize the number of open PRs and avoid redundant release/deployment cycles.



