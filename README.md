# Emergent Models

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)
- [Support](#support)

## Overview

Emergent Models is a project focused on developing and exploring emergent behaviors in machine learning models. This repository contains tools and frameworks for training, analyzing, and understanding complex model behaviors.

## Features

- [ ] Core model training framework
- [ ] Model analysis tools
- [ ] Visualization capabilities
- [ ] Experiment tracking
- [ ] Documentation and examples

## Installation

### Prerequisites
- **Python 3.13 or higher** (required)
- **Git**
- **Poetry** (recommended) or pip

### Option 1: Using Poetry (Recommended)

Poetry provides better dependency management and virtual environment handling.

#### Install Poetry First
If you don't have Poetry installed:

```bash
# On macOS/Linux
curl -sSL https://install.python-poetry.org | python3 -

# On Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -

# Alternative: using pip
pip install poetry
```

#### Install the Project
```bash
# Clone the repository
git clone https://github.com/BrightStarLabs/emergent-models.git
cd emergent-models

# Install dependencies (creates virtual environment automatically)
poetry install

# Optional: Install with specific extras
poetry install --extras "visualization jupyter"
```

### Option 2: Using pip (Alternative)

If you prefer not to use Poetry:

```bash
# Clone the repository
git clone https://github.com/BrightStarLabs/emergent-models.git
cd emergent-models

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Or install with extras
pip install -e ".[visualization,jupyter]"
```

### Python Version Compatibility

**Important**: This project requires **Python 3.13+**. If you have an older Python version:

#### Check Your Python Version
```bash
python --version
# or
python3 --version
```

#### If You Have Python < 3.13

**Option A: Install Python 3.13+ (Recommended)**
- **macOS**: Use [Homebrew](https://brew.sh/): `brew install python@3.13`
- **Windows**: Download from [python.org](https://www.python.org/downloads/)
- **Linux**: Use your package manager or [pyenv](https://github.com/pyenv/pyenv)

**Option B: Use pyenv (Cross-platform)**
```bash
# Install pyenv first (see https://github.com/pyenv/pyenv#installation)
pyenv install 3.13.0
pyenv local 3.13.0  # Use Python 3.13 for this project
```

**Option C: Use conda/mamba**
```bash
conda create -n emergent-models python=3.13
conda activate emergent-models
# Then follow installation steps above
```

### Optional Dependencies

The project includes several optional dependency groups for different use cases:

- **`dev`**: Development tools (pytest, black, mypy, etc.)
- **`docs`**: Documentation building (sphinx, themes)
- **`gpu`**: GPU acceleration (cupy)
- **`torch`**: PyTorch integration
- **`visualization`**: Enhanced visualization (seaborn, pillow, imageio)
- **`all`**: All optional dependencies

**Install specific extras:**
```bash
# With Poetry
poetry install --extras "visualization gpu"

# With pip
pip install -e ".[visualization,gpu]"
```

### Verification

Test your installation:
```bash
# With Poetry
poetry run python -c "import emergent_models; print('✅ Installation successful!')"

# With pip
python -c "import emergent_models; print('✅ Installation successful!')"
```

## Quick Start

### Basic Example

Try this and play around with the parameters:

**With Poetry:**
```bash
poetry run python examples/training_playground.py --pop-size 500 --generations 300 --batch-size 30 --elite-fraction 0.1
```

**With pip/venv:**
```bash
# Make sure your virtual environment is activated
python examples/training_playground.py --pop-size 500 --generations 300 --batch-size 30 --elite-fraction 0.1
```

You should be able to see something like this at the end of the training:

![image](https://github.com/user-attachments/assets/e07f6618-ef05-4b02-8cab-3b45e0807b65)

### Jupyter Notebook Example

For an interactive experience, check out `examples/doubling.ipynb`. The notebook provides a deeper dive into the components and step-by-step explanations.

**With Poetry:**
```bash
# Install Jupyter support
poetry install --extras jupyter

# Start Jupyter
poetry run jupyter lab examples/doubling.ipynb
```

**With pip:**
```bash
# Install Jupyter support
pip install -e ".[jupyter]"

# Start Jupyter
jupyter lab examples/doubling.ipynb
```

## Troubleshooting

### Common Issues

**"Python version not supported"**
- Ensure you have Python 3.13+: `python --version`
- Use pyenv, conda, or install Python 3.13+ from python.org

**"poetry: command not found"**
- Install Poetry: `curl -sSL https://install.python-poetry.org | python3 -`
- Or use pip alternative: `pip install poetry`
- Restart your terminal after installation

**"Module not found" errors**
- With Poetry: Make sure you're in the Poetry shell (`poetry shell`) or use `poetry run`
- With pip: Ensure your virtual environment is activated

**Import errors for optional dependencies**
- Install the required extras: `poetry install --extras "visualization"` or `pip install -e ".[visualization]"`

**Performance issues**
- Consider installing GPU acceleration: `poetry install --extras "gpu"` or `pip install -e ".[gpu]"`
- Requires CUDA-compatible GPU and drivers

## Configuration

## Development

### Setting Up Development Environment

**With Poetry (Recommended):**
```bash
# Clone and enter directory
git clone https://github.com/BrightStarLabs/emergent-models.git
cd emergent-models

# Install with development dependencies
poetry install --extras "dev docs visualization"

# Activate the virtual environment
poetry shell

# Run tests
poetry run pytest

# Format code
poetry run black .
poetry run isort .

# Type checking
poetry run mypy emergent_models
```

**With pip:**
```bash
# Clone and enter directory
git clone https://github.com/BrightStarLabs/emergent-models.git
cd emergent-models

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev,docs,visualization]"

# Run tests
pytest

# Format code
black .
isort .

# Type checking
mypy emergent_models
```

### Common Development Tasks

```bash
# Run specific tests
poetry run pytest tests/test_specific.py  # Poetry
pytest tests/test_specific.py             # pip

# Build documentation
poetry run sphinx-build docs docs/_build  # Poetry
sphinx-build docs docs/_build              # pip

# Check dependencies
poetry show --tree                         # Poetry
pip list                                   # pip
```

## Contributing

We welcome contributions! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support, please:

- Open an issue in the [GitHub repository](https://github.com/BrightStarLabs/emergent-models/issues)
- Join our [Discord community](#discord-link)
- Email info@brightstarlabs.ai
