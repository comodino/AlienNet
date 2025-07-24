#!/bin/bash

echo "🛠 Fermo eventuali container attivi..."
docker-compose down --volumes --remove-orphans

echo "🔧 Ricostruzione forzata del container (senza cache)..."
docker-compose build --no-cache

echo "🚀 Avvio del container JupyterLab su http://localhost:8888 ..."
docker-compose up
