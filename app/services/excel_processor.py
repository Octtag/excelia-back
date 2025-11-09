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
        print("\n🚀 Inicializando ExcelProcessor...")
        self.api_key = os.getenv("GOOGLE_API_KEY", "")
        self.llm = None
        self.llm_with_tools = None

        if self.api_key:
            print(f"🔑 API Key encontrada: {self.api_key[:10]}...{self.api_key[-5:]}")
            try:
                print("🤖 Inicializando Gemini Pro...")
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-pro",
                    google_api_key=self.api_key,
                    temperature=0.0
                )
                print(f"✅ Gemini inicializado correctamente")
                
                # Bind tools al LLM
                print(f"🔧 Vinculando {len(ALL_TOOLS)} herramientas al LLM...")
                self.llm_with_tools = self.llm.bind_tools(ALL_TOOLS)
                print(f"✅ Herramientas vinculadas: {[tool.name for tool in ALL_TOOLS]}")
                
            except Exception as e:
                print(f"❌ Error inicializando Gemini: {e}")
                self.llm = None
                self.llm_with_tools = None
        else:
            print("⚠️  No se encontró GOOGLE_API_KEY en variables de entorno")
            print("   → Se usará procesamiento local sin IA")

    def process_command(self, command: str, selected_cells: List[CellData]) -> str:
        """Procesa un comando sobre las celdas seleccionadas"""

        # Si hay LLM con tools configurado, usar IA
        if self.llm_with_tools:
            print("🤖 Usando IA con herramientas (Gemini + LangChain Tools)")
            try:
                return self._process_with_ai_tools(command, selected_cells)
            except Exception as e:
                print(f"❌ Error con IA, usando fallback local: {e}")
                print(f"   Tipo de error: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                return self._process_locally(command, selected_cells)

        # Fallback a procesamiento local
        print("💻 Usando procesamiento local (sin IA)")
        return self._process_locally(command, selected_cells)

    def _process_with_ai_tools(self, command: str, selected_cells: List[CellData]) -> str:
        """Procesa el comando usando Google Gemini con Tools de LangChain"""
        
        print("\n🔍 Extrayendo valores numéricos...")

        # Extraer valores numéricos de las celdas
        numbers = []
        for cell in selected_cells:
            if cell.value and cell.value.strip():
                try:
                    num = float(cell.value.strip().replace(",", "."))
                    numbers.append(num)
                    print(f"   ✓ '{cell.value}' → {num}")
                except ValueError:
                    print(f"   ✗ '{cell.value}' → No es número")
                    continue

        if not numbers:
            print("❌ No se encontraron valores numéricos")
            return "ERROR: No hay valores numéricos en las celdas seleccionadas"

        print(f"✅ Números extraídos: {numbers}")

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

        print("\n📤 Enviando prompt a Gemini...")
        print(f"   Comando: '{command}'")
        print(f"   Números: {numbers}")
        
        # Invocar el LLM con tools
        response = self.llm_with_tools.invoke([system_msg, human_msg])
        
        print("\n📥 Respuesta recibida de Gemini")
        print(f"   Tipo: {type(response)}")
        print(f"   Tool calls: {len(response.tool_calls) if response.tool_calls else 0}")

        # Verificar si el LLM quiere usar una tool
        if response.tool_calls:
            # El LLM decidió usar una tool
            tool_call = response.tool_calls[0]
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            
            print(f"\n🔧 LLM decidió usar herramienta:")
            print(f"   Herramienta: {tool_name}")
            print(f"   Argumentos: {tool_args}")

            # Buscar la tool correspondiente
            tool_to_use = None
            for tool in ALL_TOOLS:
                if tool.name == tool_name:
                    tool_to_use = tool
                    break

            if tool_to_use:
                # Si los argumentos no tienen 'numbers', usar los números extraídos
                if 'numbers' not in tool_args or not tool_args['numbers']:
                    print(f"   → Inyectando números: {numbers}")
                    tool_args['numbers'] = numbers

                # Ejecutar la tool
                print(f"   ⚙️  Ejecutando {tool_name}...")
                result = tool_to_use.invoke(tool_args)
                print(f"   ✅ Resultado: {result}")

                # Formatear resultado
                if isinstance(result, float):
                    return f"{result:.2f}"
                return str(result)
            else:
                print(f"   ❌ Herramienta '{tool_name}' no encontrada")

        # Si el LLM no usó tools, extraer el contenido
        print("\n⚠️  LLM no usó herramientas, extrayendo contenido directo")
        result = response.content.strip()
        print(f"   Contenido: {result[:100]}...")
        if result.startswith("```"):
            result = result.split("```")[1].strip()

        return result

    def _process_locally(self, command: str, selected_cells: List[CellData]) -> str:
        """Procesamiento local sin IA"""
        
        print("\n💻 Procesamiento LOCAL iniciado")
        command_lower = command.lower()
        print(f"   Comando (lowercase): '{command_lower}'")

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
            print("   ❌ No hay valores numéricos")
            return "ERROR: No hay valores numéricos en las celdas seleccionadas"
        
        print(f"   Números encontrados: {numbers}")

        # Detectar operación
        if any(word in command_lower for word in ["promedio", "media", "average", "avg"]):
            print("   🎯 Detectado: PROMEDIO")
            result = sum(numbers) / len(numbers)
            return f"{result:.2f}"

        elif any(word in command_lower for word in ["suma", "sum", "total", "sumar"]):
            print("   🎯 Detectado: SUMA")
            result = sum(numbers)
            return f"{result:.2f}"

        elif any(word in command_lower for word in ["máximo", "maximo", "max", "mayor"]):
            print("   🎯 Detectado: MÁXIMO")
            result = max(numbers)
            return f"{result:.2f}"

        elif any(word in command_lower for word in ["mínimo", "minimo", "min", "menor"]):
            print("   🎯 Detectado: MÍNIMO")
            result = min(numbers)
            return f"{result:.2f}"

        elif any(word in command_lower for word in ["count", "contar", "cantidad"]):
            print("   🎯 Detectado: CONTAR")
            return str(len(numbers))

        elif any(word in command_lower for word in ["multiplicar", "producto", "multiply"]):
            print("   🎯 Detectado: MULTIPLICAR")
            result = 1
            for num in numbers:
                result *= num
            return f"{result:.2f}"

        else:
            # Por defecto, calcular promedio
            print("   🎯 No detectado, usando: PROMEDIO (default)")
            result = sum(numbers) / len(numbers)
            return f"{result:.2f}"
