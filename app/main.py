from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

from app.models import CommandRequest, CommandResponse, TargetCell
from app.services.excel_processor import ExcelProcessor

# Cargar variables de entorno
load_dotenv()

# Inicializar FastAPI
app = FastAPI(
    title="Excelia API",
    description="Backend para Excelia - Excel con IA",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar procesador
processor = ExcelProcessor()


@app.get("/")
async def root():
    """Endpoint raíz para verificar que la API está funcionando"""
    return {
        "message": "Excelia API está funcionando",
        "version": "1.0.0",
        "ai_enabled": processor.llm is not None
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ai_enabled": processor.llm is not None
    }


@app.post("/api/excel/execute", response_model=CommandResponse)
async def execute_command(request: CommandRequest):
    """
    Ejecuta un comando sobre las celdas seleccionadas

    Args:
        request: CommandRequest con el comando y las celdas seleccionadas

    Returns:
        CommandResponse con el resultado de la operación
    """
    try:
        # Validar que hay celdas seleccionadas
        if not request.selectedCells:
            return CommandResponse(
                success=False,
                error="No se han seleccionado celdas"
            )

        # Validar que hay un comando
        if not request.command or not request.command.strip():
            return CommandResponse(
                success=False,
                error="El comando no puede estar vacío"
            )

        # Procesar el comando
        result = processor.process_command(request.command, request.selectedCells)

        # Verificar si hay error
        if result.startswith("ERROR"):
            return CommandResponse(
                success=False,
                error=result
            )

        # Retornar resultado exitoso
        return CommandResponse(
            success=True,
            result=result,
            targetCell=None  # El frontend decide dónde poner el resultado
        )

    except Exception as e:
        return CommandResponse(
            success=False,
            error=f"Error procesando comando: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True
    )
