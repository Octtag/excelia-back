from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

from app.models import (
    CommandRequest, 
    CommandResponse, 
    TargetCell,
    AskResponse,
    AnalysisMetadata,
    CalculationMetric,
    AskRequest,
    CellRange
)
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

@app.post("/api/excel/ask", response_model=CommandResponse)
async def ask_question(request: AskRequest):
    """
    Responde una pregunta del usuario sobre las celdas seleccionadas usando un agente ReAct.
    
    El agente puede:
    - Analizar los datos de las celdas seleccionadas
    - Usar herramientas para realizar cálculos y análisis estadísticos
    - Proporcionar respuestas en lenguaje natural
    - Usar historial de conversación previo
    - Sugerir cálculos después de preguntas abiertas
    
    Args:
        request: CommandRequest con la pregunta, celdas seleccionadas y historial de conversación
    
    Returns:
        CommandResponse con la respuesta del agente y sugerencias de cálculos si aplica
    """
    try:
        # Validar que hay una pregunta
        if not request.command or not request.command.strip():
            return CommandResponse(
                success=False,
                error="La pregunta no puede estar vacía"
            )
        
        # Preparar celdas seleccionadas (puede ser None o lista vacía)
        selected_cells = request.selectedCells or []
        
        # Preparar historial de conversación
        conversation_history = request.conversationHistory or []
        
        # Preparar contexto del sheet
        sheet_context = request.sheetContext or []
        
        # Procesar la pregunta con el agente ReAct (incluyendo historial y contexto del sheet)
        result = processor.answer_question(
            question=request.command,
            selected_cells=selected_cells,
            sheet_context=sheet_context,
            conversation_history=conversation_history
        )
        
        # Verificar si hay error
        if isinstance(result, str) and result.startswith("ERROR"):
            return CommandResponse(
                success=False,
                error=result
            )
        
        # Si result es un diccionario, extraer información estructurada
        if isinstance(result, dict):
            answer = result.get("answer", "")
            suggested_calculations = result.get("suggested_calculations")
            tools_used = result.get("tools_used", [])
            calculations = result.get("calculations", [])
            cells_analyzed = result.get("cells_analyzed", len(selected_cells))
            numeric_values = result.get("numeric_values", 0)
            text_values = result.get("text_values", 0)
            selectable_ranges = result.get("selectable_ranges", [])
        else:
            answer = result
            suggested_calculations = None
            tools_used = []
            calculations = []
            cells_analyzed = len(selected_cells)
            numeric_values = 0
            text_values = 0
            selectable_ranges = []
        
        # Construir metadatos del análisis
        calculation_metrics = [
            CalculationMetric(
                name=calc.get("name", ""),
                value=calc.get("value"),
                tool=calc.get("tool", "")
            )
            for calc in calculations
        ]
        
        metadata = AnalysisMetadata(
            cellsAnalyzed=cells_analyzed,
            numericValues=numeric_values,
            textValues=text_values,
            toolsUsed=tools_used,
            calculations=calculation_metrics
        )
        
        # Construir respuesta estructurada
        structured_result = AskResponse(
            answer=answer,
            summary=None,  # Se puede generar un resumen si es necesario
            metadata=metadata,
            suggestedCalculations=suggested_calculations,
            selectableRanges=selectable_ranges if selectable_ranges else None
        )
        
        # Retornar respuesta exitosa con estructura
        return CommandResponse(
            success=True,
            result=answer,  # Mantener para compatibilidad
            structuredResult=structured_result,
            targetCell=None,  # No se modifica ninguna celda, solo se responde
            suggestedCalculations=suggested_calculations  # Mantener para compatibilidad
        )
    
    except Exception as e:
        return CommandResponse(
            success=False,
            error=f"Error procesando pregunta: {str(e)}"
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
