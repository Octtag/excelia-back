from langchain_core.tools import tool
from typing import List


@tool
def calculate_average(numbers: List[float]) -> float:
    """
    Calcula el promedio (media) de una lista de números.

    Args:
        numbers: Lista de números para calcular el promedio

    Returns:
        El promedio de los números
    """
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)


@tool
def calculate_sum(numbers: List[float]) -> float:
    """
    Calcula la suma de una lista de números.

    Args:
        numbers: Lista de números para sumar

    Returns:
        La suma de los números
    """
    return sum(numbers)


@tool
def calculate_max(numbers: List[float]) -> float:
    """
    Encuentra el valor máximo en una lista de números.

    Args:
        numbers: Lista de números

    Returns:
        El valor máximo
    """
    if not numbers:
        return 0.0
    return max(numbers)


@tool
def calculate_min(numbers: List[float]) -> float:
    """
    Encuentra el valor mínimo en una lista de números.

    Args:
        numbers: Lista de números

    Returns:
        El valor mínimo
    """
    if not numbers:
        return 0.0
    return min(numbers)


@tool
def count_values(numbers: List[float]) -> int:
    """
    Cuenta la cantidad de valores en una lista.

    Args:
        numbers: Lista de números

    Returns:
        La cantidad de valores
    """
    return len(numbers)


@tool
def calculate_product(numbers: List[float]) -> float:
    """
    Calcula el producto (multiplicación) de una lista de números.

    Args:
        numbers: Lista de números para multiplicar

    Returns:
        El producto de los números
    """
    if not numbers:
        return 0.0
    result = 1.0
    for num in numbers:
        result *= num
    return result


@tool
def calculate_median(numbers: List[float]) -> float:
    """
    Calcula la mediana de una lista de números.

    Args:
        numbers: Lista de números

    Returns:
        La mediana de los números
    """
    if not numbers:
        return 0.0
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    if n % 2 == 0:
        return (sorted_numbers[n//2 - 1] + sorted_numbers[n//2]) / 2
    else:
        return sorted_numbers[n//2]


@tool
def calculate_std_deviation(numbers: List[float]) -> float:
    """
    Calcula la desviación estándar de una lista de números.

    Args:
        numbers: Lista de números

    Returns:
        La desviación estándar
    """
    if not numbers or len(numbers) < 2:
        return 0.0

    mean = sum(numbers) / len(numbers)
    variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
    return variance ** 0.5


@tool
def calculate_variance(numbers: List[float]) -> float:
    """
    Calcula la varianza de una lista de números.

    Args:
        numbers: Lista de números

    Returns:
        La varianza
    """
    if not numbers or len(numbers) < 2:
        return 0.0

    mean = sum(numbers) / len(numbers)
    return sum((x - mean) ** 2 for x in numbers) / len(numbers)


@tool
def calculate_percentage(value: float, total: float) -> float:
    """
    Calcula el porcentaje que representa un valor respecto a un total.

    Args:
        value: El valor parcial
        total: El valor total

    Returns:
        El porcentaje (0-100)
    """
    if total == 0:
        return 0.0
    return (value / total) * 100


@tool
def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calcula el cambio porcentual entre dos valores.

    Args:
        old_value: El valor anterior
        new_value: El valor nuevo

    Returns:
        El cambio porcentual (positivo si aumentó, negativo si disminuyó)
    """
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100


@tool
def count_non_empty(values: List[str]) -> int:
    """
    Cuenta cuántos valores no están vacíos en una lista.

    Args:
        values: Lista de valores (pueden ser strings o números)

    Returns:
        Cantidad de valores no vacíos
    """
    return sum(1 for v in values if v and str(v).strip())


@tool
def count_unique(values: List[str]) -> int:
    """
    Cuenta la cantidad de valores únicos en una lista.

    Args:
        values: Lista de valores

    Returns:
        Cantidad de valores únicos
    """
    return len(set(str(v) for v in values if v))


@tool
def find_duplicates(values: List[str]) -> List[str]:
    """
    Encuentra valores duplicados en una lista.

    Args:
        values: Lista de valores

    Returns:
        Lista de valores que aparecen más de una vez
    """
    seen = set()
    duplicates = set()
    for v in values:
        if v:
            v_str = str(v)
            if v_str in seen:
                duplicates.add(v_str)
            seen.add(v_str)
    return list(duplicates)


@tool
def calculate_growth_rate(values: List[float]) -> float:
    """
    Calcula la tasa de crecimiento promedio entre valores consecutivos.

    Args:
        values: Lista de números ordenados cronológicamente

    Returns:
        Tasa de crecimiento promedio en porcentaje
    """
    if len(values) < 2:
        return 0.0
    
    growth_rates = []
    for i in range(1, len(values)):
        if values[i-1] != 0:
            rate = ((values[i] - values[i-1]) / values[i-1]) * 100
            growth_rates.append(rate)
    
    return sum(growth_rates) / len(growth_rates) if growth_rates else 0.0


@tool
def calculate_cumulative_sum(numbers: List[float]) -> List[float]:
    """
    Calcula la suma acumulativa de una lista de números.

    Args:
        numbers: Lista de números

    Returns:
        Lista con las sumas acumulativas
    """
    if not numbers:
        return []
    
    cumsum = []
    total = 0.0
    for num in numbers:
        total += num
        cumsum.append(total)
    return cumsum


@tool
def calculate_range(numbers: List[float]) -> float:
    """
    Calcula el rango (diferencia entre máximo y mínimo) de una lista de números.

    Args:
        numbers: Lista de números

    Returns:
        El rango (max - min)
    """
    if not numbers:
        return 0.0
    return max(numbers) - min(numbers)


@tool
def calculate_percentile(numbers: List[float], percentile: int) -> float:
    """
    Calcula un percentil específico de una lista de números.

    Args:
        numbers: Lista de números
        percentile: El percentil a calcular (0-100)

    Returns:
        El valor del percentil especificado
    """
    if not numbers:
        return 0.0
    
    sorted_numbers = sorted(numbers)
    index = (percentile / 100) * (len(sorted_numbers) - 1)
    
    if index.is_integer():
        return sorted_numbers[int(index)]
    else:
        lower = sorted_numbers[int(index)]
        upper = sorted_numbers[int(index) + 1]
        fraction = index - int(index)
        return lower + (upper - lower) * fraction


@tool
def round_numbers(numbers: List[float], decimals: int = 2) -> List[float]:
    """
    Redondea una lista de números a una cantidad específica de decimales.

    Args:
        numbers: Lista de números
        decimals: Cantidad de decimales (por defecto 2)

    Returns:
        Lista de números redondeados
    """
    return [round(num, decimals) for num in numbers]


@tool
def calculate_weighted_average(values: List[float], weights: List[float]) -> float:
    """
    Calcula el promedio ponderado de una lista de valores con sus pesos.

    Args:
        values: Lista de valores
        weights: Lista de pesos correspondientes

    Returns:
        El promedio ponderado
    """
    if not values or not weights or len(values) != len(weights):
        return 0.0
    
    if sum(weights) == 0:
        return 0.0
    
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)


@tool
def calculate_mode(numbers: List[float]) -> float:
    """
    Encuentra el valor que más se repite (moda) en una lista de números.

    Args:
        numbers: Lista de números

    Returns:
        El valor más frecuente
    """
    if not numbers:
        return 0.0
    
    from collections import Counter
    counter = Counter(numbers)
    return counter.most_common(1)[0][0]


@tool
def filter_greater_than(numbers: List[float], threshold: float) -> List[float]:
    """
    Filtra números mayores a un umbral específico.

    Args:
        numbers: Lista de números
        threshold: Valor umbral

    Returns:
        Lista de números mayores al umbral
    """
    return [num for num in numbers if num > threshold]


@tool
def filter_less_than(numbers: List[float], threshold: float) -> List[float]:
    """
    Filtra números menores a un umbral específico.

    Args:
        numbers: Lista de números
        threshold: Valor umbral

    Returns:
        Lista de números menores al umbral
    """
    return [num for num in numbers if num < threshold]


@tool
def calculate_quartiles(numbers: List[float]) -> dict:
    """
    Calcula los cuartiles (Q1, Q2/mediana, Q3) de una lista de números.

    Args:
        numbers: Lista de números

    Returns:
        Diccionario con Q1, Q2 (mediana) y Q3
    """
    if not numbers:
        return {"Q1": 0.0, "Q2": 0.0, "Q3": 0.0}
    
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    
    # Q2 (mediana)
    if n % 2 == 0:
        q2 = (sorted_numbers[n//2 - 1] + sorted_numbers[n//2]) / 2
    else:
        q2 = sorted_numbers[n//2]
    
    # Q1 (primer cuartil)
    lower_half = sorted_numbers[:n//2]
    if len(lower_half) % 2 == 0 and lower_half:
        q1 = (lower_half[len(lower_half)//2 - 1] + lower_half[len(lower_half)//2]) / 2
    elif lower_half:
        q1 = lower_half[len(lower_half)//2]
    else:
        q1 = sorted_numbers[0]
    
    # Q3 (tercer cuartil)
    upper_half = sorted_numbers[(n+1)//2:]
    if len(upper_half) % 2 == 0 and upper_half:
        q3 = (upper_half[len(upper_half)//2 - 1] + upper_half[len(upper_half)//2]) / 2
    elif upper_half:
        q3 = upper_half[len(upper_half)//2]
    else:
        q3 = sorted_numbers[-1]
    
    return {"Q1": q1, "Q2": q2, "Q3": q3}


# Lista de todas las tools disponibles
ALL_TOOLS = [
    # Estadísticas básicas
    calculate_average,
    calculate_sum,
    calculate_max,
    calculate_min,
    count_values,
    calculate_product,
    calculate_median,
    calculate_mode,
    
    # Estadísticas avanzadas
    calculate_std_deviation,
    calculate_variance,
    calculate_range,
    calculate_percentile,
    calculate_quartiles,
    
    # Operaciones de porcentaje y crecimiento
    calculate_percentage,
    calculate_percentage_change,
    calculate_growth_rate,
    calculate_weighted_average,
    
    # Operaciones de lista
    calculate_cumulative_sum,
    round_numbers,
    filter_greater_than,
    filter_less_than,
    
    # Análisis de datos
    count_non_empty,
    count_unique,
    find_duplicates,
]
