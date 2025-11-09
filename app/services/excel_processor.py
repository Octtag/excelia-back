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
            print(f"✅ API Key encontrada: {self.api_key[:10]}...{self.api_key[-5:]}")
            try:
                print("🤖 Inicializando Gemini 2.5 Flash...")
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=self.api_key,
                    temperature=0.0
                )
                print(f"🔧 Bindeando {len(ALL_TOOLS)} herramientas al LLM...")
                # Bind tools al LLM
                self.llm_with_tools = self.llm.bind_tools(ALL_TOOLS)
                print("✅ Gemini inicializado correctamente con herramientas")
            except Exception as e:
                print(f"❌ Error inicializando Gemini: {e}")
                self.llm = None
                self.llm_with_tools = None
        else:
            print("⚠️  No se encontró GOOGLE_API_KEY, usando procesamiento local")

    def _detect_multi_column_selection(self, selected_cells: List[CellData]) -> bool:
        """Detecta si la selección tiene múltiples columnas con datos"""
        if not selected_cells:
            return False

        # Obtener todas las columnas únicas
        unique_cols = set(cell.col for cell in selected_cells)
        return len(unique_cols) > 1

    def _group_cells_by_column(self, selected_cells: List[CellData]) -> dict:
        """Agrupa las celdas por columna"""
        from collections import defaultdict
        columns = defaultdict(list)

        for cell in selected_cells:
            columns[cell.col].append(cell)

        return dict(columns)

    def process_command(self, command: str, selected_cells: List[CellData]) -> dict:
        """Procesa un comando sobre las celdas seleccionadas

        Returns:
            dict con keys: result, formula, isGeneralQuery, columnResults
        """

        print(f"\n🔄 process_command() llamado")
        print(f"   Comando: '{command}'")
        print(f"   Celdas: {len(selected_cells)}")

        # Detectar si es consulta general
        is_general = self._is_general_query(command)

        # Si es consulta general, no procesar por columnas
        if is_general:
            print("📋 Es consulta general, procesando normalmente")
            if self.llm_with_tools:
                print("🤖 Usando IA con herramientas (Gemini + LangChain Tools)")
                try:
                    result = self._process_with_ai_tools(command, selected_cells)
                    print(f"✅ IA retornó: {result}")
                    return result
                except Exception as e:
                    import traceback
                    print(f"❌ Error con IA:")
                    print(f"   Tipo: {type(e).__name__}")
                    print(f"   Mensaje: {str(e)}")
                    print(f"   Traceback:")
                    traceback.print_exc()
                    print("🔄 Intentando fallback a procesamiento local...")
                    return self._process_locally(command, selected_cells)
            else:
                print("💻 No hay IA disponible, usando procesamiento local")
                return self._process_locally(command, selected_cells)

        # Detectar si hay múltiples columnas
        has_multiple_columns = self._detect_multi_column_selection(selected_cells)

        if has_multiple_columns:
            print("📊 Detectadas múltiples columnas, procesando por columna")
            return self._process_by_columns(command, selected_cells)

        # Procesamiento normal (una sola columna o fila)
        print("📝 Procesamiento normal (columna única)")
        # Si hay LLM con tools configurado, usar IA
        if self.llm_with_tools:
            print("🤖 Usando IA con herramientas (Gemini + LangChain Tools)")
            try:
                result = self._process_with_ai_tools(command, selected_cells)
                print(f"✅ IA retornó: {result}")
                return result
            except Exception as e:
                import traceback
                print(f"❌ Error con IA:")
                print(f"   Tipo: {type(e).__name__}")
                print(f"   Mensaje: {str(e)}")
                print(f"   Traceback:")
                traceback.print_exc()
                print("🔄 Intentando fallback a procesamiento local...")
                return self._process_locally(command, selected_cells)
        else:
            print("💻 No hay IA disponible, usando procesamiento local")

        # Fallback a procesamiento local
        return self._process_locally(command, selected_cells)

    def _generate_excel_formula(self, tool_name: str, selected_cells: List[CellData]) -> str:
        """Genera una fórmula de Excel basada en la tool usada y las celdas seleccionadas"""
        # Mapeo completo de tools a funciones de Excel/HyperFormula
        tool_to_excel = {
            # Estadísticas básicas
            "calculate_sum": "SUM",
            "calculate_average": "AVERAGE",
            "calculate_max": "MAX",
            "calculate_min": "MIN",
            "count_values": "COUNT",
            "calculate_product": "PRODUCT",
            "calculate_median": "MEDIAN",
            "calculate_std_deviation": "STDEV",

            # Otras funciones estadísticas que podrían agregarse
            "calculate_variance": "VAR",
            "calculate_mode": "MODE",
            "calculate_percentile": "PERCENTILE",
            "calculate_quartile": "QUARTILE",
            "calculate_rank": "RANK",
            "calculate_correl": "CORREL",
            "calculate_covariance": "COVAR",

            # Funciones matemáticas
            "calculate_abs": "ABS",
            "calculate_sqrt": "SQRT",
            "calculate_power": "POWER",
            "calculate_round": "ROUND",
            "calculate_floor": "FLOOR",
            "calculate_ceiling": "CEILING",

            # Funciones de conteo
            "count_if": "COUNTIF",
            "count_blank": "COUNTBLANK",
            "count_a": "COUNTA",
        }

        excel_func = tool_to_excel.get(tool_name)
        if not excel_func:
            return None

        # Obtener el rango de celdas
        if not selected_cells:
            return None

        # Convertir índices a notación Excel (A1, B2, etc.)
        def col_to_letter(col):
            result = ""
            col += 1  # Excel es 1-indexed
            while col > 0:
                col -= 1
                result = chr(65 + (col % 26)) + result
                col //= 26
            return result

        # Generar referencia de rango
        first_cell = selected_cells[0]
        last_cell = selected_cells[-1]

        start_ref = f"{col_to_letter(first_cell.col)}{first_cell.row + 1}"
        end_ref = f"{col_to_letter(last_cell.col)}{last_cell.row + 1}"

        # Si es una sola celda
        if start_ref == end_ref:
            formula = f"={excel_func}({start_ref})"
        else:
            formula = f"={excel_func}({start_ref}:{end_ref})"

        return formula

    def _is_general_query(self, command: str) -> bool:
        """Detecta si el comando es una consulta general/interpretación sin cálculo"""

        command_lower = command.lower().strip()

        # Primero verificar si contiene palabras de cálculo (prioridad alta)
        calculation_keywords = [
            "calcula", "suma", "sumar", "promedio", "media", "average",
            "máximo", "maximo", "max", "mayor", "mínimo", "minimo", "min", "menor",
            "cuenta", "contar", "count", "cantidad",
            "multiplica", "multiplicar", "producto", "multiply",
            "divide", "dividir", "division",
            "mediana", "median",
            "desviación", "desviacion", "stdev", "std",
            "varianza", "variance", "var",
            "total", "sum"
        ]

        # Si tiene palabra de cálculo, NO es consulta general
        if any(keyword in command_lower for keyword in calculation_keywords):
            return False

        # Solo marcar como general si tiene EXPLÍCITAMENTE palabras interrogativas o de análisis
        # Y debe ser una pregunta o solicitud de interpretación clara
        general_patterns = [
            "qué significa", "que significa",
            "qué representa", "que representa",
            "qué quiere decir", "que quiere decir",
            "cómo interpretar", "como interpretar",
            "explica", "explicar", "explicame", "explícame",
            "interpreta", "interpretar",
            "analiza", "analizar",
            "describe", "describir",
            "por qué", "porque", "porqué",
            "ayuda con", "ayúdame",
            "qué es esto", "que es esto",
            "qué son", "que son"
        ]

        # Solo es consulta general si contiene uno de estos patrones específicos
        return any(pattern in command_lower for pattern in general_patterns)

    def _process_with_ai_tools(self, command: str, selected_cells: List[CellData]) -> dict:
        """Procesa el comando usando Google Gemini con Tools de LangChain

        Returns:
            dict con keys: result, formula, isGeneralQuery
        """

        print("\n🔬 _process_with_ai_tools() iniciado")

        # Detectar si es consulta general
        is_general = self._is_general_query(command)
        print(f"   ¿Es consulta general?: {is_general}")

        # Extraer valores numéricos de las celdas
        numbers = []
        print(f"📊 Extrayendo números de {len(selected_cells)} celdas...")
        for i, cell in enumerate(selected_cells):
            if cell.value and cell.value.strip():
                try:
                    num = float(cell.value.strip().replace(",", "."))
                    numbers.append(num)
                    if i < 5:  # Mostrar solo los primeros 5
                        print(f"   ✓ Celda [{cell.row},{cell.col}]: '{cell.value}' → {num}")
                except ValueError:
                    if i < 5:
                        print(f"   ✗ Celda [{cell.row},{cell.col}]: '{cell.value}' (no numérico)")
                    continue

        print(f"📈 Números extraídos: {numbers[:10]}{'...' if len(numbers) > 10 else ''}")
        
        if not numbers:
            print("❌ No se encontraron valores numéricos")
            return "ERROR: No hay valores numéricos en las celdas seleccionadas"

        # Construir contexto de las celdas
        cells_context = "\n".join([
            f"Fila {cell.row + 1}, Columna {cell.col + 1}: {cell.value if cell.value else '(vacío)'}"
            for cell in selected_cells
        ])

        # Crear prompt que instruye al LLM a usar tools
        print("\n📝 Creando prompt para Gemini...")
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

        print(f"📤 Enviando prompt a Gemini...")
        print(f"   System message: {len(system_msg.content)} chars")
        print(f"   Human message: {len(human_msg.content)} chars")
        
        # Invocar el LLM con tools
        response = self.llm_with_tools.invoke([system_msg, human_msg])
        print(f"📥 Respuesta recibida de Gemini")

        # Verificar si el LLM quiere usar una tool
        if response.tool_calls:
            print(f"🔧 Gemini decidió usar {len(response.tool_calls)} herramienta(s)")
            # El LLM decidió usar una tool
            tool_call = response.tool_calls[0]
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            
            print(f"   Tool seleccionada: {tool_name}")
            print(f"   Argumentos: {tool_args}")

            # Buscar la tool correspondiente
            tool_to_use = None
            for tool in ALL_TOOLS:
                if tool.name == tool_name:
                    tool_to_use = tool
                    break

            if tool_to_use:
                print(f"✅ Tool encontrada: {tool_name}")
                # Si los argumentos no tienen 'numbers', usar los números extraídos
                if 'numbers' not in tool_args or not tool_args['numbers']:
                    print(f"   📊 Inyectando números extraídos: {numbers}")
                    tool_args['numbers'] = numbers

                # Ejecutar la tool
                print(f"⚙️  Ejecutando tool '{tool_name}'...")
                result = tool_to_use.invoke(tool_args)
                print(f"✅ Tool ejecutada, resultado: {result}")

                # Generar fórmula de Excel
                formula = self._generate_excel_formula(tool_name, selected_cells)
                print(f"📐 Fórmula generada: {formula}")

                # Formatear resultado
                if isinstance(result, float):
                    formatted = f"{result:.2f}"
                    print(f"📊 Resultado formateado: {formatted}")
                    return {
                        "result": formatted,
                        "formula": formula,
                        "isGeneralQuery": is_general
                    }
                print(f"📊 Resultado como string: {str(result)}")
                return {
                    "result": str(result),
                    "formula": formula,
                    "isGeneralQuery": is_general
                }
            else:
                print(f"❌ Tool '{tool_name}' no encontrada en ALL_TOOLS")
        else:
            print("ℹ️  Gemini no usó herramientas, extrayendo contenido directo")

        # Si el LLM no usó tools, es probablemente una consulta general
        result = response.content.strip()
        print(f"📄 Contenido de respuesta: '{result[:100]}...'")

        if result.startswith("```"):
            result = result.split("```")[1].strip()
            print(f"📄 Contenido limpiado de markdown: '{result}'")

        # Retornar como consulta general (sin fórmula)
        return {
            "result": result,
            "formula": None,
            "isGeneralQuery": True
        }

    def _process_locally(self, command: str, selected_cells: List[CellData]) -> dict:
        """Procesamiento local sin IA

        Returns:
            dict con keys: result, formula, isGeneralQuery
        """

        print("\n💻 _process_locally() iniciado")
        print(f"   Comando: '{command}'")

        command_lower = command.lower()
        print(f"   Comando (lowercase): '{command_lower}'")

        # Detectar si es consulta general
        is_general = self._is_general_query(command)

        # Extraer valores numéricos
        numbers = []
        print(f"📊 Extrayendo números localmente de {len(selected_cells)} celdas...")
        for i, cell in enumerate(selected_cells):
            if cell.value and cell.value.strip():
                try:
                    # Intentar parsear como número
                    num = float(cell.value.strip().replace(",", "."))
                    numbers.append(num)
                    if i < 5:
                        print(f"   ✓ '{cell.value}' → {num}")
                except ValueError:
                    if i < 5:
                        print(f"   ✗ '{cell.value}' (no numérico)")
                    continue

        print(f"📈 Números extraídos: {numbers}")

        if not numbers:
            print("❌ No se encontraron valores numéricos")
            return {
                "result": "ERROR: No hay valores numéricos en las celdas seleccionadas",
                "formula": None,
                "isGeneralQuery": is_general
            }

        # Detectar operación y generar fórmula
        print("🔍 Detectando operación...")
        tool_name = None
        result_value = None

        if any(word in command_lower for word in ["promedio", "media", "average", "avg"]):
            print("   → Operación: PROMEDIO")
            tool_name = "calculate_average"
            result_value = sum(numbers) / len(numbers)

        elif any(word in command_lower for word in ["suma", "sum", "total", "sumar"]):
            print("   → Operación: SUMA")
            tool_name = "calculate_sum"
            result_value = sum(numbers)

        elif any(word in command_lower for word in ["máximo", "maximo", "max", "mayor"]):
            print("   → Operación: MÁXIMO")
            tool_name = "calculate_max"
            result_value = max(numbers)

        elif any(word in command_lower for word in ["mínimo", "minimo", "min", "menor"]):
            print("   → Operación: MÍNIMO")
            tool_name = "calculate_min"
            result_value = min(numbers)

        elif any(word in command_lower for word in ["count", "contar", "cantidad"]):
            print("   → Operación: CONTAR")
            tool_name = "count_values"
            result_value = len(numbers)

        elif any(word in command_lower for word in ["multiplicar", "producto", "multiply"]):
            print("   → Operación: PRODUCTO")
            tool_name = "calculate_product"
            result_value = 1
            for num in numbers:
                result_value *= num

        else:
            print("   → Operación: PROMEDIO (por defecto)")
            # Por defecto, calcular promedio
            tool_name = "calculate_average"
            result_value = sum(numbers) / len(numbers)

        # Generar fórmula
        formula = self._generate_excel_formula(tool_name, selected_cells) if tool_name else None

        # Formatear resultado
        if isinstance(result_value, float):
            result_str = f"{result_value:.2f}"
        else:
            result_str = str(result_value)

        return {
            "result": result_str,
            "formula": formula,
            "isGeneralQuery": is_general
        }

    def _process_by_columns(self, command: str, selected_cells: List[CellData]) -> dict:
        """Procesa el comando por cada columna individualmente

        Returns:
            dict con keys: columnResults (lista de resultados por columna)
        """
        print("\n📊 _process_by_columns() iniciado")

        # Agrupar celdas por columna
        columns = self._group_cells_by_column(selected_cells)
        print(f"   Columnas detectadas: {sorted(columns.keys())}")

        column_results = []

        for col_index in sorted(columns.keys()):
            col_cells = columns[col_index]
            print(f"\n   📍 Procesando columna {col_index} con {len(col_cells)} celdas")

            # Procesar esta columna (usar procesamiento local para cada una)
            result_dict = self._process_locally(command, col_cells)

            if not result_dict.get("result", "").startswith("ERROR"):
                column_results.append({
                    "col": col_index,
                    "result": result_dict.get("result"),
                    "formula": result_dict.get("formula")
                })
                print(f"   ✅ Columna {col_index}: {result_dict.get('result')}")
            else:
                print(f"   ⚠️  Columna {col_index}: Sin datos numéricos")

        if not column_results:
            return {
                "result": "ERROR: No hay valores numéricos en ninguna columna",
                "formula": None,
                "isGeneralQuery": False
            }

        print(f"\n✅ Procesadas {len(column_results)} columnas exitosamente")

        return {
            "columnResults": column_results,
            "isGeneralQuery": False
        }
