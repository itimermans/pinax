# pinax

Pinax is a modular Python library for extracting tabular data from various sources and performing structured analysis. It is designed for extensibility, allowing users to add new data sources and analysis methods easily.

## Features
- Extract tabular data from files and locations into a common format
- Perform structured analysis on tables
- Modular and extensible architecture

## Installation
```bash
pip install -e .
```
Requires Python >=3.8. All dependencies are managed via `pyproject.toml`.

## Usage Example
```python
import pinax
# Add your data extraction and analysis code here
```

## Project Structure
- `src/pinax/`: Main library code
- `src/pinax/utils.py`: Utility functions
- `tests/`: Test suite (uses pytest)
- `.vscode/`: Editor and debug configuration
- `pyproject.toml`: Project metadata and dependencies

## Development
- Install dev dependencies:
	```bash
	pip install -e .[dev]
	```
- Run tests:
	```bash
	pytest
	```
- Format code:
	```bash
	black src/ tests/
	```
- Lint code:
	```bash
	flake8 src/ tests/
	```

## Contributing
- Add new features as modules/functions in `src/pinax/`
- Write tests in `tests/`
- Ensure code passes formatting and linting

## License
MIT License. See `LICENSE` for details.