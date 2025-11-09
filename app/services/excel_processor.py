import os
import re
from typing import List, Optional, TypedDict, Annotated, Sequence
from app.models import CellData, CellRange
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from app.services.excel_tools import ALL_TOOLS
from langgraph.graph import StateGraph, END
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

    def _filter_tools_by_question(self, question: str, selected_cells: List[CellData]) -> List:
        """
        Filtra herramientas dinámicamente basándose en la pregunta y contexto.
        
        Args:
            question: La pregunta del usuario
            selected_cells: Lista de celdas seleccionadas
            
        Returns:
            Lista filtrada de herramientas relevantes
        """
        question_lower = question.lower()
        
        # Extraer números para determinar si necesitamos herramientas numéricas
        has_numbers = False
        for cell in selected_cells:
            if cell.value and cell.value.strip():
                try:
                    float(cell.value.strip().replace(",", "."))
                    has_numbers = True
                    break
                except ValueError:
                    pass
        
        # Si no hay números, solo usar herramientas de texto
        if not has_numbers:
            text_tools = [
                tool for tool in ALL_TOOLS 
                if tool.name in ['count_non_empty', 'count_unique', 'find_duplicates']
            ]
            return text_tools if text_tools else ALL_TOOLS
        
        # Filtrar herramientas basándose en palabras clave de la pregunta
        filtered_tools = []
        
        # Palabras clave para diferentes categorías de herramientas
        keywords = {
            'statistics': ['promedio', 'media', 'average', 'mediana', 'median', 'desviación', 'deviation', 'varianza', 'variance', 'estadística', 'statistic'],
            'basic_math': ['suma', 'sum', 'total', 'máximo', 'maximo', 'max', 'mínimo', 'minimo', 'min', 'producto', 'product'],
            'advanced': ['percentil', 'percentile', 'cuartil', 'quartile', 'crecimiento', 'growth', 'acumulativo', 'cumulative'],
            'filtering': ['filtrar', 'filter', 'mayor que', 'greater than', 'menor que', 'less than'],
            'counting': ['contar', 'count', 'cantidad', 'cuántos', 'how many', 'únicos', 'unique', 'duplicados', 'duplicates']
        }
        
        # Determinar qué categorías son relevantes
        relevant_categories = []
        for category, words in keywords.items():
            if any(word in question_lower for word in words):
                relevant_categories.append(category)
        
        # Si no hay categorías relevantes, usar todas las herramientas
        if not relevant_categories:
            return ALL_TOOLS
        
        # Mapear categorías a herramientas
        category_tools = {
            'statistics': ['calculate_average', 'calculate_median', 'calculate_std_deviation', 'calculate_variance', 'calculate_mode'],
            'basic_math': ['calculate_sum', 'calculate_max', 'calculate_min', 'calculate_product', 'calculate_range'],
            'advanced': ['calculate_percentile', 'calculate_quartiles', 'calculate_growth_rate', 'calculate_cumulative_sum', 'calculate_weighted_average'],
            'filtering': ['filter_greater_than', 'filter_less_than'],
            'counting': ['count_values', 'count_non_empty', 'count_unique', 'find_duplicates']
        }
        
        # Recopilar herramientas relevantes
        tool_names = set()
        for category in relevant_categories:
            if category in category_tools:
                tool_names.update(category_tools[category])
        
        # Filtrar herramientas
        filtered_tools = [tool for tool in ALL_TOOLS if tool.name in tool_names]
        
        # Si no se encontraron herramientas específicas, usar todas
        return filtered_tools if filtered_tools else ALL_TOOLS

    def answer_question(
        self, 
        question: str, 
        selected_cells: Optional[List[CellData]] = None,
        sheet_context: Optional[List[CellData]] = None,
        conversation_history: Optional[List] = None,
        tools: Optional[List] = None, 
        auto_filter: bool = True
    ) -> dict:
        """
        Responde una pregunta del usuario sobre las celdas seleccionadas usando un agente ReAct.
        
        El agente puede:
        - Analizar los datos de las celdas
        - Usar herramientas para realizar cálculos
        - Proporcionar respuestas en lenguaje natural
        - Usar historial de conversación previo
        - Sugerir cálculos después de preguntas abiertas
        
        Args:
            question: La pregunta del usuario
            selected_cells: Lista de celdas seleccionadas (opcional, puede ser None o lista vacía)
            sheet_context: Contexto completo del sheet (todas las celdas de la hoja)
            conversation_history: Historial de conversación previo (lista de dicts con 'question' y 'answer')
            tools: Lista opcional de herramientas a usar. Si es None, usa todas las herramientas disponibles.
            auto_filter: Si es True, filtra herramientas automáticamente basándose en la pregunta. Si es False, usa todas las herramientas.
        
        Returns:
            dict con 'answer' y opcionalmente 'suggested_calculations'
        """
        # Normalizar selected_cells (puede ser None)
        selected_cells = selected_cells or []
        
        if not self.llm:
            answer = self._answer_question_fallback(question, selected_cells)
            return {"answer": answer}
        
        # Si no se especifican herramientas, usar todas o filtrar automáticamente
        if tools is None:
            if auto_filter:
                tools = self._filter_tools_by_question(question, selected_cells)
            else:
                tools = ALL_TOOLS
        
        # Preparar historial de conversación
        history = conversation_history or []
        
        # Preparar contexto del sheet
        sheet_ctx = sheet_context or []
        
        try:
            result_data = self._answer_question_with_react(question, selected_cells, tools, history, sheet_ctx)
            
            # Extraer información del resultado
            if isinstance(result_data, dict):
                answer = result_data.get("answer", "")
                tools_used = result_data.get("tools_used", [])
                calculations = result_data.get("calculations", [])
            else:
                answer = result_data
                tools_used = []
                calculations = []
            
            # Detectar si la pregunta es abierta y generar sugerencias
            is_open_question = self._is_open_question(question)
            suggested_calculations = None
            
            if is_open_question and len(history) == 0:  # Solo sugerir después de la primera pregunta abierta
                suggested_calculations = self._suggest_calculations(selected_cells, answer)
            
            # Contar valores numéricos y de texto
            numbers = []
            text_values = []
            for cell in selected_cells:
                if cell.value and cell.value.strip():
                    try:
                        float(cell.value.strip().replace(",", "."))
                        numbers.append(cell.value)
                    except ValueError:
                        text_values.append(cell.value)
            
            return {
                "answer": answer,
                "suggested_calculations": suggested_calculations,
                "tools_used": tools_used,
                "calculations": calculations,
                "cells_analyzed": len(selected_cells),
                "numeric_values": len(numbers),
                "text_values": len(text_values)
            }
        except Exception as e:
            print(f"Error con agente ReAct, usando fallback: {e}")
            answer = self._answer_question_fallback(question, selected_cells)
            return {"answer": answer}

    def _answer_question_with_react(
        self, 
        question: str, 
        selected_cells: List[CellData], 
        tools: List,
        conversation_history: Optional[List] = None,
        sheet_context: Optional[List[CellData]] = None
    ) -> dict:
        """Implementa un agente ReAct usando LangGraph para responder preguntas con herramientas dinámicas"""
        
        # Normalizar selected_cells (puede ser None o lista vacía)
        selected_cells = selected_cells or []
        
        # Construir contexto detallado de las celdas seleccionadas
        cells_data = []
        numbers = []
        all_values = []
        
        for cell in selected_cells:
            cell_info = {
                "row": cell.row + 1,  # 1-indexed para el usuario
                "col": cell.col + 1,  # 1-indexed para el usuario
                "value": cell.value if cell.value else "(vacío)"
            }
            cells_data.append(cell_info)
            all_values.append(cell.value if cell.value else "")
            
            # Extraer números
            if cell.value and cell.value.strip():
                try:
                    num = float(cell.value.strip().replace(",", "."))
                    numbers.append(num)
                except ValueError:
                    pass
        
        # Construir contexto del sheet completo
        sheet_data = []
        sheet_numbers = []
        sheet_all_values = []
        
        if sheet_context:
            for cell in sheet_context:
                cell_info = {
                    "row": cell.row + 1,  # 1-indexed para el usuario
                    "col": cell.col + 1,  # 1-indexed para el usuario
                    "value": cell.value if cell.value else "(vacío)"
                }
                sheet_data.append(cell_info)
                sheet_all_values.append(cell.value if cell.value else "")
                
                # Extraer números del sheet
                if cell.value and cell.value.strip():
                    try:
                        num = float(cell.value.strip().replace(",", "."))
                        sheet_numbers.append(num)
                    except ValueError:
                        pass
        
        # Las herramientas se usarán dinámicamente desde el estado
        
        # Definir el estado del agente
        class AgentState(TypedDict):
            messages: Annotated[Sequence[BaseMessage], "Lista de mensajes en la conversación"]
            question: str
            cells_context: str
            sheet_context: str
            numbers: List[float]
            all_values: List[str]
            iterations: int
            available_tools: List  # Herramientas disponibles dinámicamente (renombrado para evitar conflicto con LangGraph)
        
        # Crear el grafo del agente
        workflow = StateGraph(AgentState)
        
        # Rastrear herramientas y cálculos utilizados
        tools_used_tracker = []
        calculations_tracker = []
        
        # Nodo de herramienta dinámico - las herramientas vienen del estado
        def execute_tools(state: AgentState) -> AgentState:
            """Ejecuta herramientas dinámicamente basándose en el estado"""
            messages = state["messages"]
            available_tools = state.get("available_tools", tools)
            
            # Crear diccionario de herramientas por nombre
            tools_dict = {tool.name: tool for tool in available_tools}
            
            # Obtener el último mensaje que debería tener tool calls
            last_message = messages[-1]
            
            if not isinstance(last_message, AIMessage) or not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
                return state
            
            # Ejecutar cada tool call
            tool_messages = []
            for tool_call in last_message.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                
                if tool_name in tools_dict:
                    tool = tools_dict[tool_name]
                    try:
                        # Ejecutar la herramienta
                        result = tool.invoke(tool_args)
                        
                        # Rastrear herramienta utilizada
                        if tool_name not in tools_used_tracker:
                            tools_used_tracker.append(tool_name)
                        
                        # Rastrear cálculo realizado
                        calculation_name = self._get_calculation_name(tool_name)
                        calculations_tracker.append({
                            "name": calculation_name,
                            "value": result,
                            "tool": tool_name
                        })
                        
                        # Crear mensaje de herramienta
                        tool_message = ToolMessage(
                            content=str(result),
                            tool_call_id=tool_call.get('id', tool_name)
                        )
                        tool_messages.append(tool_message)
                    except Exception as e:
                        # Si hay error, crear mensaje de error
                        error_message = ToolMessage(
                            content=f"Error ejecutando {tool_name}: {str(e)}",
                            tool_call_id=tool_call.get('id', tool_name)
                        )
                        tool_messages.append(error_message)
            
            # Agregar mensajes de herramientas al estado
            return {
                **state,
                "messages": list(messages) + tool_messages
            }
        
        def should_continue(state: AgentState) -> str:
            """Decide si continuar con herramientas o finalizar"""
            messages = state["messages"]
            if not messages:
                return "end"
            
            last_message = messages[-1]
            
            # Si hay llamadas a herramientas, ejecutarlas
            if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"
            
            # Si no hay más iteraciones, finalizar
            if state["iterations"] >= 10:  # Límite de iteraciones
                return "end"
            
            # Si el último mensaje no tiene tool calls, es una respuesta final
            return "end"
        
        def call_model(state: AgentState) -> AgentState:
            """Llama al modelo LLM con herramientas dinámicas"""
            messages = state["messages"]
            available_tools = state.get("available_tools", tools)
            
            # Crear LLM con herramientas dinámicas del estado
            llm_with_state_tools = self.llm.bind_tools(available_tools)
            
            # Construir lista de herramientas disponibles para el prompt
            tools_list = "\n".join([
                f"- {tool.name}: {tool.description}" for tool in available_tools
            ])
            
            # Construir el prompt del sistema con contexto mejorado
            cells_context_str = state.get("cells_context", "")
            sheet_context_str = state.get("sheet_context", "")
            
            # Construir sección de celdas seleccionadas (solo si hay celdas)
            cells_section = ""
            if cells_context_str and cells_context_str.strip():
                cells_section = f"""
═══════════════════════════════════════════════════════════════
CELDAS SELECCIONADAS (foco principal del análisis):
═══════════════════════════════════════════════════════════════
{cells_context_str}

VALORES NUMÉRICOS EN CELDAS SELECCIONADAS: {state["numbers"]}
TODOS LOS VALORES EN CELDAS SELECCIONADAS: {state["all_values"]}"""
            
            # Construir sección del sheet (solo si hay contexto del sheet)
            sheet_context_section = ""
            sheet_instructions = ""
            
            if sheet_context_str and sheet_context_str.strip():
                sheet_context_section = f"""
═══════════════════════════════════════════════════════════════
CONTEXTO COMPLETO DEL SHEET (para referencia y comparación):
═══════════════════════════════════════════════════════════════
{sheet_context_str}
═══════════════════════════════════════════════════════════════

IMPORTANTE: Usa este contexto del sheet completo para:
- Comparar las celdas seleccionadas con el resto del sheet
- Entender la estructura general de los datos
- Proporcionar contexto adicional sobre relaciones entre datos
- Identificar patrones o tendencias en todo el sheet
- Hacer análisis más informados basados en el contexto completo
"""
                sheet_instructions = """
7. Si tienes contexto del sheet completo, úsalo ACTIVAMENTE para:
   - Comparar las celdas seleccionadas con el resto del sheet
   - Proporcionar contexto adicional sobre cómo se relacionan los datos
   - Identificar si las celdas seleccionadas son representativas del sheet completo
   - Mencionar patrones o tendencias que observes en todo el sheet
   - Hacer análisis comparativos cuando sea relevante"""
            
            system_prompt = f"""Eres un asistente experto en análisis de datos de hojas de cálculo Excel.
Tienes acceso a herramientas (tools) para realizar cálculos y análisis estadísticos.
{cells_section}{sheet_context_section}
═══════════════════════════════════════════════════════════════
HERRAMIENTAS DISPONIBLES:
═══════════════════════════════════════════════════════════════
{tools_list}

═══════════════════════════════════════════════════════════════
INSTRUCCIONES:
═══════════════════════════════════════════════════════════════
1. Analiza la pregunta del usuario cuidadosamente
2. Si necesitas hacer cálculos, USA las herramientas disponibles (NO calcules tú mismo)
3. Puedes usar múltiples herramientas si es necesario para responder completamente
4. Después de obtener resultados de las herramientas, proporciona una respuesta clara y completa en lenguaje natural
5. Incluye los valores numéricos relevantes en tu respuesta
6. Si la pregunta no requiere cálculos, responde directamente basándote en el contexto disponible{sheet_instructions}

═══════════════════════════════════════════════════════════════
TABLAS EN MARKDOWN:
═══════════════════════════════════════════════════════════════
Cuando el usuario solicite una tabla o cuando sea apropiado mostrar datos estructurados, USA tablas en formato Markdown.

FORMATO DE TABLAS MARKDOWN:
| Columna 1 | Columna 2 | Columna 3 |
|-----------|-----------|-----------|
| Valor 1   | Valor 2   | Valor 3   |
| Valor 4   | Valor 5   | Valor 6   |

REGLAS PARA TABLAS:
1. SIEMPRE incluye la fila de encabezado con los nombres de las columnas
2. SIEMPRE incluye la fila separadora con guiones (---) después del encabezado
3. Alinea el contenido de las celdas de manera legible
4. Usa tablas cuando:
   - El usuario lo solicite explícitamente ("muéstrame en una tabla", "haz una tabla", "formato tabla")
   - Tengas datos comparativos (ej: resultados por canal, métricas por categoría)
   - Necesites mostrar múltiples valores relacionados (ej: estadísticas, resúmenes)
5. Las tablas deben ser claras y fáciles de leer

EJEMPLOS DE CUANDO USAR TABLAS:
- "Muestra los resultados en una tabla" → Crea tabla con los datos
- "Compara los canales" → Crea tabla comparativa
- "Resumen de métricas" → Crea tabla con métricas
- "Lista de valores" → Considera usar tabla si hay múltiples columnas

EJEMPLO DE TABLA CORRECTA:
| Canal | Impresiones | Clics | Conversiones | ROAS |
|-------|-------------|-------|--------------|------|
| Google Ads | 100000 | 5000 | 250 | 3.5 |
| Facebook | 80000 | 4000 | 200 | 2.8 |
| Instagram | 60000 | 3000 | 150 | 2.2 |

═══════════════════════════════════════════════════════════════
REGLAS OBLIGATORIAS PARA REFERENCIAS A CELDAS:
═══════════════════════════════════════════════════════════════
IMPORTANTE: SIEMPRE que te refieras a un rango específico de celdas, DEBES usar la etiqueta <selectRange>.

NUNCA uses referencias ambiguas como:
- ❌ "INPUTS (Filas 1-6 y 11-16):"
- ❌ "Las celdas en las filas 10-15"
- ❌ "Los datos en la columna A, filas 5-10"

SIEMPRE usa la etiqueta <selectRange> con el formato correcto:
- ✅ "Puedes ver los datos en <selectRange startRow="0" startCol="0" endRow="5" endCol="16">esta sección</selectRange>"
- ✅ "Los valores más altos están en <selectRange startRow="9" startCol="0" endRow="14" endCol="0">estas celdas</selectRange>"
- ✅ "Los INPUTS están en <selectRange startRow="0" startCol="0" endRow="5" endCol="10" label="INPUTS">esta sección</selectRange>"

FORMATO OBLIGATORIO DE LA ETIQUETA:
<selectRange startRow="FILA_INICIO" startCol="COL_INICIO" endRow="FILA_FIN" endCol="COL_FIN" label="TEXTO_OPCIONAL">TEXTO_VISIBLE</selectRange>

REGLAS:
1. Las coordenadas son 0-indexed (fila 0 = primera fila, columna 0 = primera columna)
2. SIEMPRE incluye la etiqueta cuando menciones rangos de celdas específicos
3. El atributo "label" es opcional pero recomendado para claridad
4. El texto visible entre las etiquetas es lo que el usuario verá como hipervínculo
5. Si mencionas múltiples rangos, usa una etiqueta <selectRange> para cada uno

EJEMPLOS CORRECTOS:
- "Los datos de entrada están en <selectRange startRow="0" startCol="0" endRow="5" endCol="10">esta sección</selectRange>"
- "Puedes ver los resultados en <selectRange startRow="10" startCol="0" endRow="15" endCol="5" label="Resultados">estas celdas</selectRange>"
- "Los INPUTS (Filas 1-6) están en <selectRange startRow="0" startCol="0" endRow="5" endCol="16">esta sección</selectRange>"

RECUERDA: Es OBLIGATORIO usar <selectRange> cuando te refieras a rangos de celdas. NO uses referencias de texto plano."""
            
            system_msg = SystemMessage(content=system_prompt)
            
            # Agregar mensajes del sistema y humano
            full_messages = [system_msg] + list(messages)
            
            # Llamar al modelo con herramientas dinámicas
            response = llm_with_state_tools.invoke(full_messages)
            
            # Preservar todo el estado y agregar la nueva respuesta
            return {
                **state,
                "messages": list(messages) + [response],
                "iterations": state["iterations"] + 1
            }
        
        # Agregar nodos al grafo
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", execute_tools)
        
        # Definir el punto de entrada
        workflow.set_entry_point("agent")
        
        # Agregar condicional para decidir si usar herramientas o finalizar
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                "end": END
            }
        )
        
        # Después de usar herramientas, volver al agente
        workflow.add_edge("tools", "agent")
        
        # Compilar el grafo
        app = workflow.compile()
        
        # Preparar el estado inicial
        cells_context = ""
        if cells_data:
            cells_context = "\n".join([
                f"Fila {cell['row']}, Columna {cell['col']}: {cell['value']}"
                for cell in cells_data
            ])
        
        # Preparar contexto del sheet con mejor formato y estadísticas
        sheet_context_str = ""
        sheet_stats = {}
        
        if sheet_data:
            # Calcular estadísticas del sheet
            sheet_rows = set(cell['row'] for cell in sheet_data)
            sheet_cols = set(cell['col'] for cell in sheet_data)
            sheet_stats = {
                "total_cells": len(sheet_data),
                "total_rows": len(sheet_rows),
                "total_cols": len(sheet_cols),
                "numeric_values": len(sheet_numbers),
                "text_values": len(sheet_all_values) - len(sheet_numbers),
                "min_row": min(sheet_rows) if sheet_rows else 0,
                "max_row": max(sheet_rows) if sheet_rows else 0,
                "min_col": min(sheet_cols) if sheet_cols else 0,
                "max_col": max(sheet_cols) if sheet_cols else 0
            }
            
            # Formatear contexto del sheet de manera más inteligente
            if len(sheet_data) <= 200:
                # Si hay pocas celdas, mostrar todas organizadas por filas
                sheet_context_str = self._format_sheet_context(sheet_data, sheet_stats)
            else:
                # Si hay muchas celdas, mostrar un resumen inteligente
                sheet_context_str = self._format_sheet_summary(sheet_data, sheet_stats)
        
        # Construir mensajes iniciales con historial de conversación
        initial_messages = []
        
        # Agregar historial de conversación previo
        if conversation_history:
            for msg in conversation_history:
                # msg puede ser un dict con 'question' y 'answer' o un objeto ConversationMessage
                if isinstance(msg, dict):
                    question_hist = msg.get('question', '')
                    answer_hist = msg.get('answer', '')
                else:
                    question_hist = getattr(msg, 'question', '')
                    answer_hist = getattr(msg, 'answer', '')
                
                if question_hist:
                    initial_messages.append(HumanMessage(content=question_hist))
                if answer_hist:
                    initial_messages.append(AIMessage(content=answer_hist))
        
        # Agregar la pregunta actual
        initial_messages.append(HumanMessage(content=question))
        
        initial_state = {
            "messages": initial_messages,
            "question": question,
            "cells_context": cells_context,
            "sheet_context": sheet_context_str,
            "numbers": numbers,
            "all_values": all_values,
            "iterations": 0,
            "available_tools": tools  # Herramientas dinámicas en el estado
        }
        
        # Ejecutar el agente
        final_state = app.invoke(initial_state)
        
        # Extraer la respuesta final
        messages = final_state["messages"]
        answer = None
        
        # Buscar la última respuesta del agente que no tenga tool calls (respuesta final)
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                # Si no tiene tool_calls o tiene tool_calls vacío, es una respuesta final
                if not hasattr(message, 'tool_calls') or not message.tool_calls:
                    if message.content and message.content.strip():
                        answer = message.content.strip()
                        break
        
        # Si no hay respuesta clara del agente, buscar el último mensaje de herramienta
        if not answer:
            for message in reversed(messages):
                if isinstance(message, ToolMessage):
                    answer = f"Resultado del análisis: {message.content}"
                    break
        
        # Si no hay nada, retornar mensaje de error
        if not answer:
            answer = "No se pudo generar una respuesta. Por favor, intenta reformular tu pregunta."
        
        # Extraer rangos seleccionables de la respuesta (etiquetas <selectRange>)
        selectable_ranges = self._extract_selectable_ranges(answer)
        
        # Debug: imprimir si se encontraron rangos
        if selectable_ranges:
            print(f"Se encontraron {len(selectable_ranges)} rangos seleccionables")
            for r in selectable_ranges:
                print(f"  - Rango: fila {r.startRow}-{r.endRow}, col {r.startCol}-{r.endCol}, label: {r.label}")
        
        # Retornar información estructurada
        return {
            "answer": answer,
            "tools_used": tools_used_tracker,
            "calculations": calculations_tracker,
            "selectable_ranges": selectable_ranges
        }

    def _answer_question_fallback(self, question: str, selected_cells: List[CellData]) -> str:
        """Fallback simple sin IA para responder preguntas básicas"""
        question_lower = question.lower()
        
        # Extraer valores numéricos
        numbers = []
        for cell in selected_cells:
            if cell.value and cell.value.strip():
                try:
                    num = float(cell.value.strip().replace(",", "."))
                    numbers.append(num)
                except ValueError:
                    pass
        
        # Respuestas simples basadas en palabras clave
        if not numbers:
            return f"Las celdas seleccionadas contienen: {', '.join([c.value if c.value else '(vacío)' for c in selected_cells[:5]])}"
        
        if any(word in question_lower for word in ["promedio", "media", "average"]):
            avg = sum(numbers) / len(numbers)
            return f"El promedio de los valores seleccionados es {avg:.2f}"
        
        if any(word in question_lower for word in ["suma", "sum", "total"]):
            total = sum(numbers)
            return f"La suma de los valores seleccionados es {total:.2f}"
        
        if any(word in question_lower for word in ["máximo", "maximo", "max", "mayor"]):
            max_val = max(numbers)
            return f"El valor máximo es {max_val:.2f}"
        
        if any(word in question_lower for word in ["mínimo", "minimo", "min", "menor"]):
            min_val = min(numbers)
            return f"El valor mínimo es {min_val:.2f}"
        
        # Respuesta genérica
        return f"Hay {len(numbers)} valores numéricos en las celdas seleccionadas. Los valores son: {', '.join([str(n) for n in numbers[:10]])}"

    def _is_open_question(self, question: str) -> bool:
        """
        Detecta si una pregunta es abierta (no específica sobre un cálculo).
        
        Las preguntas abiertas son aquellas que no solicitan un cálculo específico,
        como "¿Qué puedes decirme sobre estos datos?" o "¿Qué análisis puedo hacer?"
        """
        question_lower = question.lower()
        
        # Palabras clave que indican preguntas abiertas
        open_question_keywords = [
            "qué", "que", "what", "cómo", "como", "how",
            "cuéntame", "cuentame", "tell me", "dime", "explícame", "explicame", "explain",
            "análisis", "analisis", "analysis", "analizar", "analyze",
            "información", "informacion", "information", "datos", "data",
            "puedo hacer", "can i", "sugerencias", "suggestions",
            "recomendaciones", "recommendations", "opciones", "options"
        ]
        
        # Palabras clave que indican preguntas específicas (NO abiertas)
        specific_question_keywords = [
            "promedio", "media", "average", "suma", "sum", "total",
            "máximo", "maximo", "max", "mínimo", "minimo", "min",
            "mediana", "median", "desviación", "deviation", "varianza", "variance",
            "percentil", "percentile", "cuartil", "quartile", "rango", "range",
            "cuántos", "cuantos", "how many", "cuántas", "cuantas"
        ]
        
        # Si contiene palabras de preguntas específicas, NO es abierta
        if any(keyword in question_lower for keyword in specific_question_keywords):
            return False
        
        # Si contiene palabras de preguntas abiertas, es abierta
        if any(keyword in question_lower for keyword in open_question_keywords):
            return True
        
        # Si la pregunta es muy corta o genérica, considerarla abierta
        if len(question.split()) <= 3:
            return True
        
        return False

    def _suggest_calculations(self, selected_cells: List[CellData], current_answer: str) -> List[str]:
        """
        Genera sugerencias de cálculos basándose en las celdas seleccionadas y la respuesta actual.
        
        Args:
            selected_cells: Lista de celdas seleccionadas
            current_answer: Respuesta actual del agente
            
        Returns:
            Lista de sugerencias de cálculos en lenguaje natural
        """
        suggestions = []
        
        # Extraer números de las celdas
        numbers = []
        for cell in selected_cells:
            if cell.value and cell.value.strip():
                try:
                    num = float(cell.value.strip().replace(",", "."))
                    numbers.append(num)
                except ValueError:
                    pass
        
        # Si no hay números, sugerir análisis de texto
        if not numbers or len(numbers) < 2:
            if len(selected_cells) > 0:
                suggestions.append("Contar cuántos valores no están vacíos")
                suggestions.append("Contar valores únicos")
                suggestions.append("Encontrar valores duplicados")
            return suggestions
        
        # Sugerencias básicas para datos numéricos
        if len(numbers) >= 2:
            suggestions.append("Calcular el promedio de los valores")
            suggestions.append("Calcular la suma total")
            suggestions.append("Encontrar el valor máximo")
            suggestions.append("Encontrar el valor mínimo")
        
        # Sugerencias estadísticas avanzadas
        if len(numbers) >= 3:
            suggestions.append("Calcular la mediana")
            suggestions.append("Calcular la desviación estándar")
            suggestions.append("Calcular el rango (diferencia entre máximo y mínimo)")
        
        # Sugerencias para series temporales o secuenciales
        if len(numbers) >= 4:
            suggestions.append("Calcular la tasa de crecimiento")
            suggestions.append("Calcular la suma acumulativa")
            suggestions.append("Calcular los cuartiles (Q1, Q2, Q3)")
        
        # Sugerencias avanzadas
        if len(numbers) >= 5:
            suggestions.append("Calcular el percentil 75")
            suggestions.append("Calcular la varianza")
            suggestions.append("Identificar valores atípicos")
        
        # Limitar a 5-6 sugerencias más relevantes
        return suggestions[:6]

    def _get_calculation_name(self, tool_name: str) -> str:
        """Obtiene un nombre legible para un cálculo basado en el nombre de la herramienta"""
        name_mapping = {
            "calculate_average": "Promedio",
            "calculate_sum": "Suma",
            "calculate_max": "Valor máximo",
            "calculate_min": "Valor mínimo",
            "calculate_median": "Mediana",
            "calculate_std_deviation": "Desviación estándar",
            "calculate_variance": "Varianza",
            "calculate_range": "Rango",
            "calculate_percentile": "Percentil",
            "calculate_quartiles": "Cuartiles",
            "calculate_growth_rate": "Tasa de crecimiento",
            "calculate_cumulative_sum": "Suma acumulativa",
            "calculate_product": "Producto",
            "calculate_mode": "Moda",
            "count_values": "Conteo de valores",
            "count_non_empty": "Conteo de valores no vacíos",
            "count_unique": "Conteo de valores únicos",
            "find_duplicates": "Valores duplicados"
        }
        return name_mapping.get(tool_name, tool_name.replace("_", " ").title())

    def _format_sheet_context(self, sheet_data: List[dict], stats: dict) -> str:
        """Formatea el contexto del sheet de manera organizada por filas"""
        if not sheet_data:
            return ""
        
        # Organizar celdas por fila
        rows_dict = {}
        for cell in sheet_data:
            row = cell['row']
            if row not in rows_dict:
                rows_dict[row] = []
            rows_dict[row].append(cell)
        
        # Ordenar filas
        sorted_rows = sorted(rows_dict.keys())
        
        # Construir contexto formateado
        context_parts = [
            f"ESTADÍSTICAS DEL SHEET:",
            f"- Total de celdas: {stats.get('total_cells', 0)}",
            f"- Filas: {stats.get('total_rows', 0)} (desde fila {stats.get('min_row', 0)} hasta {stats.get('max_row', 0)})",
            f"- Columnas: {stats.get('total_cols', 0)} (desde columna {stats.get('min_col', 0)} hasta {stats.get('max_col', 0)})",
            f"- Valores numéricos: {stats.get('numeric_values', 0)}",
            f"- Valores de texto: {stats.get('text_values', 0)}",
            "",
            "DATOS DEL SHEET (organizados por filas):"
        ]
        
        # Agregar datos por fila
        for row in sorted_rows[:100]:  # Limitar a 100 filas
            cells_in_row = sorted(rows_dict[row], key=lambda x: x['col'])
            row_data = ", ".join([f"Col {c['col']}: {c['value']}" for c in cells_in_row])
            context_parts.append(f"Fila {row}: {row_data}")
        
        if len(sorted_rows) > 100:
            context_parts.append(f"... (mostrando primeras 100 filas de {len(sorted_rows)} totales)")
        
        return "\n".join(context_parts)

    def _format_sheet_summary(self, sheet_data: List[dict], stats: dict) -> str:
        """Formatea un resumen inteligente del sheet cuando hay muchas celdas"""
        if not sheet_data:
            return ""
        
        # Organizar por filas para el resumen
        rows_dict = {}
        for cell in sheet_data:
            row = cell['row']
            if row not in rows_dict:
                rows_dict[row] = []
            rows_dict[row].append(cell)
        
        sorted_rows = sorted(rows_dict.keys())
        
        # Obtener primeras y últimas filas
        first_rows = sorted_rows[:25]
        last_rows = sorted_rows[-25:] if len(sorted_rows) > 50 else []
        
        context_parts = [
            f"ESTADÍSTICAS DEL SHEET:",
            f"- Total de celdas: {stats.get('total_cells', 0)}",
            f"- Filas: {stats.get('total_rows', 0)} (desde fila {stats.get('min_row', 0)} hasta {stats.get('max_row', 0)})",
            f"- Columnas: {stats.get('total_cols', 0)} (desde columna {stats.get('min_col', 0)} hasta {stats.get('max_col', 0)})",
            f"- Valores numéricos: {stats.get('numeric_values', 0)}",
            f"- Valores de texto: {stats.get('text_values', 0)}",
            "",
            f"RESUMEN DEL SHEET (mostrando primeras 25 filas y últimas 25 filas de {len(sorted_rows)} totales):",
            "",
            "PRIMERAS FILAS:"
        ]
        
        # Agregar primeras filas
        for row in first_rows:
            cells_in_row = sorted(rows_dict[row], key=lambda x: x['col'])
            row_data = ", ".join([f"Col {c['col']}: {c['value']}" for c in cells_in_row[:10]])  # Limitar columnas por fila
            if len(cells_in_row) > 10:
                row_data += f" ... (+{len(cells_in_row) - 10} columnas más)"
            context_parts.append(f"Fila {row}: {row_data}")
        
        if last_rows:
            context_parts.append("")
            context_parts.append("ÚLTIMAS FILAS:")
            for row in last_rows:
                cells_in_row = sorted(rows_dict[row], key=lambda x: x['col'])
                row_data = ", ".join([f"Col {c['col']}: {c['value']}" for c in cells_in_row[:10]])
                if len(cells_in_row) > 10:
                    row_data += f" ... (+{len(cells_in_row) - 10} columnas más)"
                context_parts.append(f"Fila {row}: {row_data}")
        
        context_parts.append("")
        context_parts.append(f"NOTA: El sheet completo contiene {stats.get('total_cells', 0)} celdas. "
                            f"Usa este contexto para entender la estructura general y hacer comparaciones.")
        
        return "\n".join(context_parts)

    def _extract_selectable_ranges(self, answer: str) -> List[CellRange]:
        """
        Extrae etiquetas <selectRange> de la respuesta y las convierte en objetos CellRange.
        
        Formato esperado: <selectRange startRow="0" startCol="0" endRow="5" endCol="2">Texto del enlace</selectRange>
        
        Args:
            answer: Texto de la respuesta que puede contener etiquetas <selectRange>
            
        Returns:
            Lista de objetos CellRange extraídos de la respuesta
        """
        ranges = []
        
        # Patrón regex mejorado para encontrar etiquetas <selectRange>
        # Formato 1: <selectRange startRow="0" startCol="0" endRow="5" endCol="2">texto</selectRange>
        # Formato 2: <selectRange startRow="0" startCol="0" endRow="5" endCol="2" label="texto">texto</selectRange>
        # El regex debe capturar: startRow, startCol, endRow, endCol, label (opcional), contenido
        
        # Patrón más flexible que maneja ambos formatos
        pattern = r'<selectRange\s+startRow=["\']?(\d+)["\']?\s+startCol=["\']?(\d+)["\']?\s+endRow=["\']?(\d+)["\']?\s+endCol=["\']?(\d+)["\']?(?:\s+label=["\']([^"\']+)["\'])?\s*>([^<]*)</selectRange>'
        
        matches = re.finditer(pattern, answer, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            try:
                start_row = int(match.group(1))
                start_col = int(match.group(2))
                end_row = int(match.group(3))
                end_col = int(match.group(4))
                
                # label puede estar en el grupo 5 (atributo label) o grupo 6 (contenido)
                label_attr = match.group(5) if match.group(5) else None
                content = match.group(6).strip() if match.group(6) else ""
                
                # Priorizar label del atributo, si no existe usar el contenido
                label = label_attr if label_attr else content
                
                # Validar que el rango sea válido
                if start_row <= end_row and start_col <= end_col:
                    ranges.append(CellRange(
                        startRow=start_row,
                        startCol=start_col,
                        endRow=end_row,
                        endCol=end_col,
                        label=label if label else None
                    ))
            except (ValueError, IndexError, AttributeError) as e:
                # Si hay error parseando, continuar con el siguiente match
                print(f"Error parseando selectRange: {e}")
                continue
        
        return ranges
