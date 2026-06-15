FROM python:3.12-slim

WORKDIR /app

# Install system requirements and Python dependencies.
RUN python -m pip install --upgrade pip

COPY pyproject.toml uv.lock ./
RUN python -m pip install uv
RUN python -m pip install .

COPY . .

ENTRYPOINT ["uv", "run", "python", "-m", "bellman_equation"]
