set shell := ["bash", "-c"]
set windows-shell := ["powershell.exe", "-c"]

[private]
default:
  @just --list

# Install the project
@install:
  poetry install

# Check for lint errors
@lint:
  poetry run ruff check .
  poetry run ruff format --diff .

# Fix lint errors automatically
@lint-fix:
  poetry run ruff check --fix .
  poetry run ruff format .

# Check for type hint errors
@typecheck:
  poetry run mypy --check .

# Run tests
@test:
  poetry run pytest
