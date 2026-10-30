class DimensionalityError(Exception):
    pass

def validate_dimensionality(data):
    if data.ndim != 2 or data.shape[1] != 2:
        raise DimensionalityError(
            f'Некорректная размерность данных! Должна быть n x 2, получена {data.shape}'
        )
