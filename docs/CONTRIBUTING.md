Development Workflow

- Use Python 3.10+ and create a virtual environment.
- Install dev deps: pip install -r requirements-dev.txt
- Run tests and linters before pushing: make test lint typecheck
- For new features, include unit tests and, when applicable, an e2e smoke test.

Style and Type Hints

- Follow ruff/black/isort defaults; keep functions small and well-named.
- Add docstrings (numpydoc style preferred) for public APIs.

Security and Licensing

- Avoid unsafe deserialization; prefer json/orjson.
- Run make audit and resolve HIGH/CRITICAL findings.
- Record third-party licenses via make licenses for releases.

Reproducibility

- Ensure new experiments can be run via Makefile tasks.
- Update docs/REPRODUCE.md for any new figures/tables.


