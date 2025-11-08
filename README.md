# Excelia Backend - Python FastAPI

Backend moderno en Python usando FastAPI, LangChain y Google Gemini para procesar comandos de Excel con IA.

## Características

- **FastAPI**: Framework web moderno y rápido
- **LangChain**: Para orquestar llamadas a LLMs
- **LangChain Tools**: Herramientas matemáticas que el LLM puede usar
- **Google Gemini**: IA para interpretar comandos y decidir qué tool usar
- **Arquitectura Tool-Based**: El LLM NO hace los cálculos, usa las tools
- **Procesamiento local**: Fallback sin necesidad de API key
- **CORS habilitado**: Para desarrollo con frontend

## Cómo Funciona (Con IA)

1. El usuario escribe un comando en lenguaje natural (ej: "Calcula el promedio")
2. El comando se envía al LLM (Gemini) junto con las celdas seleccionadas
3. El LLM **decide qué herramienta (tool) usar** según el comando
4. El LLM **llama a la tool apropiada** con los valores numéricos
5. La **tool ejecuta el cálculo** (no el LLM)
6. El resultado se devuelve al usuario

**Ejemplo de flujo:**
```
Comando: "Calcula el promedio de estos valores"
→ LLM decide: usar tool "calculate_average"
→ Tool recibe: [10, 20, 30]
→ Tool calcula: sum([10,20,30]) / 3 = 20.0
→ Resultado: "20.00"
```

## Tools Disponibles

El LLM puede usar las siguientes herramientas para hacer cálculos:

- `calculate_average`: Calcula el promedio
- `calculate_sum`: Suma los valores
- `calculate_max`: Encuentra el máximo
- `calculate_min`: Encuentra el mínimo
- `count_values`: Cuenta los valores
- `calculate_product`: Multiplica los valores
- `calculate_median`: Calcula la mediana
- `calculate_std_deviation`: Calcula la desviación estándar

## Instalación

### 1. Crear entorno virtual

```bash
cd excelia-backend-py
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Configurar variables de entorno (Opcional)

```bash
cp .env.example .env
```

Edita `.env` y agrega tu API key de Google Gemini:
```
GOOGLE_API_KEY=tu-api-key-aqui
```

**Nota:** Si no configuras la API key, el backend funcionará con procesamiento local.

## Uso

### Iniciar el servidor

```bash
# Con reload automático (desarrollo)
python -m app.main

# O usando uvicorn directamente
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

El servidor estará disponible en `http://localhost:8000`

### Endpoints

- `GET /` - Información de la API
- `GET /health` - Health check
- `POST /api/excel/execute` - Ejecutar comando sobre celdas

### Ejemplo de request

```bash
curl -X POST http://localhost:8000/api/excel/execute \
  -H "Content-Type: application/json" \
  -d '{
    "command": "Calcula el promedio",
    "selectedCells": [
      {"row": 0, "col": 0, "value": "10"},
      {"row": 0, "col": 1, "value": "20"},
      {"row": 0, "col": 2, "value": "30"}
    ]
  }'
```

### Ejemplo de response

```json
{
  "success": true,
  "result": "20.00",
  "targetCell": null,
  "error": null
}
```

## Comandos Soportados (Modo Local)

- Calcular promedio / media
- Suma / total
- Máximo / mayor
- Mínimo / menor
- Contar valores
- Multiplicar / producto

Con Google Gemini habilitado, puedes usar comandos más naturales y complejos.

## Documentación Interactiva

Una vez que el servidor esté corriendo, visita:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Estructura del Proyecto

```
excelia-backend-py/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app y endpoints
│   ├── models.py            # Modelos Pydantic
│   └── services/
│       ├── __init__.py
│       └── excel_processor.py  # Lógica de procesamiento
├── .env                     # Variables de entorno (no commitear)
├── .env.example            # Ejemplo de variables
├── .gitignore
├── requirements.txt        # Dependencias
└── README.md
```

## Desarrollo

### Agregar más funcionalidades

Para agregar nuevas operaciones, edita `app/services/excel_processor.py` en el método `_process_locally()`.

### Integrar con LangGraph

Para workflows más complejos, puedes usar LangGraph en `excel_processor.py`:

```python
from langgraph.graph import StateGraph

# Definir estados y transiciones para workflows complejos
```

## Producción

Para producción, considera:

1. Usar variables de entorno seguras
2. Configurar CORS con dominios específicos
3. Agregar autenticación
4. Usar gunicorn o similar:

```bash
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```
