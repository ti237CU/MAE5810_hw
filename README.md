Requirements

This package requires:

Python 3.12 or newer

NumPy 2.5.3 or newer

Matplotlib 3.11.1 or newer

PyQt5 5.15.11 or newer

The project uses uv for package management and building.

Installation

Recommended: Using uv

If uv is not already installed, install it by following the instructions at:

https://docs.astral.sh/uv/

Clone or download this repository, then open a terminal in the project directory.

cd mae5810-code

Install Python and all project dependencies:

uv sync

uv sync reads the dependencies from pyproject.toml and creates the project's virtual environment automatically.

Running the Package

The project defines the following command:

[project.scripts]
mae5810-code = "mae5810_code:main"

Run the package with:

uv run mae5810-code

This executes the main() function in the mae5810_code package.

Running Individual Python Files

To run an individual Python script in the project, use:

uv run python path/to/script.py

For example:

uv run python examples/example.py

Replace path/to/script.py with the path to the Python file you want to run.

Dependencies

The required Python packages are defined in pyproject.toml:

dependencies = [
    "matplotlib>=3.11.1",
    "numpy>=2.5.3",
    "pyqt5>=5.15.11",
]

You normally do not need to install these packages individually when using uv sync.

Alternative Installation Using pip

If you do not want to use uv, create and activate a Python virtual environment first.

macOS / Linux

python3.12 -m venv .venv
source .venv/bin/activate

Windows

py -3.12 -m venv .venv
.venv\Scripts\activate

Then install the project and its dependencies:

pip install -e .

After installation, run:

mae5810-code

Project Configuration

The package configuration is stored in:

pyproject.toml

The project requires Python 3.12 or newer:

requires-python = ">=3.12"

The build system uses:

[build-system]
requires = ["uv_build>=0.12.5,<0.13.0"]
build-backend = "uv_build"
