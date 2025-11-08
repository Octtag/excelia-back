#!/bin/bash

echo "🚀 Iniciando Excelia Backend Python (FastAPI + LangChain)..."
echo "El servidor estará disponible en http://localhost:8000"
echo "Documentación: http://localhost:8000/docs"
echo ""

# Activar entorno virtual si existe
if [ -d "venv" ]; then
    echo "Activando entorno virtual..."
    source venv/bin/activate
fi

# Iniciar el servidor
python -m app.main
