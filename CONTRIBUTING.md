# Contributing

Focused bug reports, numerical reproductions, and pull requests are welcome.

1. Create a Python 3.13 virtual environment.
2. Install the project with `python -m pip install -e .[dev]`.
3. Add or update tests for every behavioral change.
4. Run `python -m pytest -q` before opening a pull request.
5. Keep numerical claims tied to reproducible inputs, tolerances, and error metrics.

Please open an issue before proposing a large API change or adding a new solver family. The public repository intentionally contains published and release-ready methods only.
