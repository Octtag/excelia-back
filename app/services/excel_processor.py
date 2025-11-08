import os
from typing import List, Optional
from app.models import CellData
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from app.services.excel_tools import ALL_TOOLS
import json


class ExcelProcessor:
    """Procesador de comandos para celdas de Excel usando LangChain, Gemini y Tools"""

    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY", "")
        self.llm = None
        self.llm_with_tools = None

        if self.api_key:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-pro",
                    google_api_key=self.api_key,
                    temperature=0.0
                )
                # Bind tools al LLM
                self.llm_with_tools = self.llm.bind_tools(ALL_TOOLS)
            except Exception as e:
                print(f"Error inicializando Gemini: {e}")
                self.llm = None
                self.llm_with_tools = None

    def process_command(self, command: str, selected_cells: List[CellData]) -> str:
        """Procesa un comando sobre las celdas seleccionadas"""

        # Si hay LLM con tools configurado, usar IA
        if self.llm_with_tools:
            try:
                return self._process_with_ai_tools(command, selected_cells)
            except Exception as e:
                print(f"Error con IA, usando fallback local: {e}")
                return self._process_locally(command, selected_cells)

        # Fallback a procesamiento local
        return self._process_locally(command, selected_cells)

    def _process_with_ai_tools(self, command: str, selected_cells: List[CellData]) -> str:
        """Procesa el comando usando Google Gemini con Tools de LangChain"""

        # Extraer valores numéricos de las celdas
        numbers = []
        for cell in selected_cells:
            if cell.value and cell.value.strip():
                try:
                    num = float(cell.value.strip().replace(",", "."))
                    numbers.append(num)
                except ValueError:
                    continue

        if not numbers:
            return "ERROR: No hay valores numéricos en las celdas seleccionadas"

        # Construir contexto de las celdas
        cells_context = "\n".join([
            f"Fila {cell.row + 1}, Columna {cell.col + 1}: {cell.value if cell.value else '(vacío)'}"
            for cell in selected_cells
        ])

        # Crear prompt que instruye al LLM a usar tools
        system_msg = SystemMessage(content="""Eres un asistente para una hoja de cálculo tipo Excel.
Tienes acceso a herramientas (tools) para realizar cálculos matemáticos.

IMPORTANTE:
- DEBES usar las herramientas disponibles para hacer los cálculos, NO calcules tú mismo.
- Lee el comando del usuario y decide qué herramienta usar.
- Llama a la herramienta apropiada con la lista de números extraída.
- Las herramientas disponibles son: calculate_average, calculate_sum, calculate_max, calculate_min, count_values, calculate_product, calculate_median, calculate_std_deviation.
- Después de llamar a la herramienta, devuelve el resultado con formato de 2 decimales.""")

        human_msg = HumanMessage(content=f"""Celdas seleccionadas:
{cells_context}

Valores numéricos extraídos: {numbers}

Comando del usuario: {command}

Usa la herramienta apropiada para ejecutar este comando sobre los valores numéricos.""")

        # Invocar el LLM con tools
        response = self.llm_with_tools.invoke([system_msg, human_msg])

        # Verificar si el LLM quiere usar una tool
        if response.tool_calls:
            # El LLM decidió usar una tool
            tool_call = response.tool_calls[0]
            tool_name = tool_call['name']
            tool_args = tool_call['args']

            # Buscar la tool correspondiente
            tool_to_use = None
            for tool in ALL_TOOLS:
                if tool.name == tool_name:
                    tool_to_use = tool
                    break

            if tool_to_use:
                # Si los argumentos no tienen 'numbers', usar los números extraídos
                if 'numbers' not in tool_args or not tool_args['numbers']:
                    tool_args['numbers'] = numbers

                # Ejecutar la tool
                result = tool_to_use.invoke(tool_args)

                # Formatear resultado
                if isinstance(result, float):
                    return f"{result:.2f}"
                return str(result)

        # Si el LLM no usó tools, extraer el contenido
        result = response.content.strip()
        if result.startswith("```"):
            result = result.split("```")[1].strip()

        return result

    def _process_locally(self, command: str, selected_cells: List[CellData]) -> str:
        """Procesamiento local sin IA"""

        command_lower = command.lower()

        # Extraer valores numéricos
        numbers = []
        for cell in selected_cells:
            if cell.value and cell.value.strip():
                try:
                    # Intentar parsear como número
                    num = float(cell.value.strip().replace(",", "."))
                    numbers.append(num)
                except ValueError:
                    continue

        if not numbers:
            return "ERROR: No hay valores numéricos en las celdas seleccionadas"

        # Detectar operación
        if any(word in command_lower for word in ["promedio", "media", "average", "avg"]):
            result = sum(numbers) / len(numbers)
            return f"{result:.2f}"

        elif any(word in command_lower for word in ["suma", "sum", "total", "sumar"]):
            result = sum(numbers)
            return f"{result:.2f}"

        elif any(word in command_lower for word in ["máximo", "maximo", "max", "mayor"]):
            result = max(numbers)
            return f"{result:.2f}"

        elif any(word in command_lower for word in ["mínimo", "minimo", "min", "menor"]):
            result = min(numbers)
            return f"{result:.2f}"

        elif any(word in command_lower for word in ["count", "contar", "cantidad"]):
            return str(len(numbers))

        elif any(word in command_lower for word in ["multiplicar", "producto", "multiply"]):
            result = 1
            for num in numbers:
                result *= num
            return f"{result:.2f}"

        else:
            # Por defecto, calcular promedio
            result = sum(numbers) / len(numbers)
            return f"{result:.2f}"
