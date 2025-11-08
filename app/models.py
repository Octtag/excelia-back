from pydantic import BaseModel, Field
from typing import List, Optional


class CellData(BaseModel):
    row: int = Field(..., description="Fila de la celda (0-indexed)")
    col: int = Field(..., description="Columna de la celda (0-indexed)")
    value: str = Field(..., description="Valor de la celda")


class CommandRequest(BaseModel):
    command: str = Field(..., description="Comando en lenguaje natural")
    selectedCells: List[CellData] = Field(..., description="Celdas seleccionadas")


class TargetCell(BaseModel):
    row: int
    col: int


class CommandResponse(BaseModel):
    success: bool
    result: Optional[str] = None
    targetCell: Optional[TargetCell] = None
    error: Optional[str] = None
