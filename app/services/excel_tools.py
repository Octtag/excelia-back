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


# Lista de todas las tools disponibles
ALL_TOOLS = [
    calculate_average,
    calculate_sum,
    calculate_max,
    calculate_min,
    count_values,
    calculate_product,
    calculate_median,
    calculate_std_deviation,
]
