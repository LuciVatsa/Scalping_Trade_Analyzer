#!/bin/bash
# Creates the full project directory structure

mkdir -p .github/workflows
mkdir -p backend/app/{api/v1/endpoints,core/{strategies,engines,models},services,websockets,db/repositories,utils,middleware,workers}
mkdir -p backend/tests/{unit,integration}
mkdir -p frontend/src/{components/{common,charts,trading,layout},pages,hooks,store/slices,services,types,utils,styles}
mkdir -p frontend/tests/{components,hooks}
mkdir -p shared/{types,constants,utils}
mkdir -p infrastructure/{docker,k8s,terraform,monitoring}
mkdir -p {docs,scripts,tests}

# Create initial files
touch backend/app/main.py
touch backend/pyproject.toml
touch frontend/package.json
touch docker-compose.yml
touch Makefile
touch .env.example
echo "Created project structure."