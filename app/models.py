from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class CellData(BaseModel):
    row: int = Field(..., description="Fila de la celda (0-indexed)")
    col: int = Field(..., description="Columna de la celda (0-indexed)")
    value: str = Field(..., description="Valor de la celda")


class ConversationMessage(BaseModel):
    """Mensaje en el historial de conversación"""
    question: str = Field(..., description="Pregunta del usuario")
    answer: str = Field(..., description="Respuesta del agente")


class CommandRequest(BaseModel):
    command: str = Field(..., description="Comando en lenguaje natural")
    selectedCells: List[CellData] = Field(..., description="Celdas seleccionadas")
    sheetContext: Optional[List[CellData]] = Field(
        default=None,
        description="Contexto completo del sheet (todas las celdas de la hoja)"
    )
    conversationHistory: Optional[List[ConversationMessage]] = Field(
        default=None,
        description="Historial de conversación previa (preguntas y respuestas anteriores)"
    )


class AskRequest(BaseModel):
    """Request específico para el endpoint /api/excel/ask que permite selectedCells como opcional"""
    command: str = Field(..., description="Pregunta en lenguaje natural")
    selectedCells: Optional[List[CellData]] = Field(
        default=None,
        description="Celdas seleccionadas (opcional, puede ser null)"
    )
    sheetContext: Optional[List[CellData]] = Field(
        default=None,
        description="Contexto completo del sheet (todas las celdas de la hoja)"
    )
    conversationHistory: Optional[List[ConversationMessage]] = Field(
        default=None, 
        description="Historial de conversación previa (preguntas y respuestas anteriores)"
    )


class TargetCell(BaseModel):
    row: int
    col: int


class ColumnResult(BaseModel):
    """Resultado para una columna específica"""
    col: int  # Índice de columna
    result: str  # Resultado calculado
    formula: str  # Fórmula de Excel
class CellRange(BaseModel):
    """Rango de celdas para selección"""
    startRow: int = Field(..., description="Fila inicial (0-indexed)")
    startCol: int = Field(..., description="Columna inicial (0-indexed)")
    endRow: int = Field(..., description="Fila final (0-indexed)")
    endCol: int = Field(..., description="Columna final (0-indexed)")
    label: Optional[str] = Field(None, description="Texto del enlace/hipervínculo")


class CalculationMetric(BaseModel):
    """Métrica calculada por una herramienta"""
    name: str = Field(..., description="Nombre de la métrica (ej: 'promedio', 'suma')")
    value: Any = Field(..., description="Valor calculado")
    tool: str = Field(..., description="Herramienta utilizada para calcular")
    unit: Optional[str] = Field(None, description="Unidad de medida si aplica")


class AnalysisMetadata(BaseModel):
    """Metadatos del análisis realizado"""
    cellsAnalyzed: int = Field(..., description="Número de celdas analizadas")
    numericValues: int = Field(..., description="Número de valores numéricos encontrados")
    textValues: int = Field(..., description="Número de valores de texto encontrados")
    toolsUsed: List[str] = Field(default_factory=list, description="Lista de herramientas utilizadas")
    calculations: List[CalculationMetric] = Field(
        default_factory=list, 
        description="Métricas calculadas durante el análisis"
    )


class AskResponse(BaseModel):
    """Respuesta estructurada del servicio ask"""
    answer: str = Field(..., description="Respuesta principal en lenguaje natural (puede contener etiquetas <selectRange>)")
    summary: Optional[str] = Field(None, description="Resumen breve de la respuesta")
    metadata: Optional[AnalysisMetadata] = Field(None, description="Metadatos del análisis")
    suggestedCalculations: Optional[List[str]] = Field(
        default=None,
        description="Sugerencias de cálculos a realizar después de una pregunta abierta"
    )
    selectableRanges: Optional[List[CellRange]] = Field(
        default=None,
        description="Rangos de celdas que pueden ser seleccionados mediante etiquetas <selectRange> en la respuesta"
    )


class CommandResponse(BaseModel):
    success: bool
    result: Optional[str] = None
    formula: Optional[str] = None  # Fórmula de Excel a insertar
    isGeneralQuery: bool = False  # True si es consulta general/interpretación
    targetCell: Optional[TargetCell] = None
    error: Optional[str] = None
    columnResults: Optional[List[ColumnResult]] = None  # Resultados por columna
    result: Optional[str] = None  # Mantener para compatibilidad
    structuredResult: Optional[AskResponse] = Field(
        None,
        description="Respuesta estructurada del servicio ask"
    )
    targetCell: Optional[TargetCell] = None
    error: Optional[str] = None
    suggestedCalculations: Optional[List[str]] = Field(
        default=None,
        description="Sugerencias de cálculos (mantener para compatibilidad)"
    )
