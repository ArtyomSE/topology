class DimensionalityError(Exception):
    pass

def validate_data_dimensionality(data):
    if data.shape[1] != 2:
        raise IncorrectDimensionalityError(
            'DimensionalityError: the dimensionality of input data is incorrect! (should be 2)'
        )