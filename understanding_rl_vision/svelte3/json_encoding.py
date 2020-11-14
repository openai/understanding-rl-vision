import numpy as np
import json


def maybe_round(x, *, precision):
    if isinstance(x, list):
        return [maybe_round(y, precision=precision) for y in x]
    else:
        if precision is None:
            return float(x)
        else:
            return float(
                np.format_float_positional(
                    x, precision=precision, unique=False, fractional=False
                )
            )


def encoder(precision=None):
    class CustomJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (tuple, set)):
                return list(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return maybe_round(obj, precision=precision)
            elif isinstance(obj, np.ndarray):
                return maybe_round(obj.tolist(), precision=precision)
            elif hasattr(obj, "to_json"):
                return obj.to_json()
            return json.JSONEncoder.default(self, obj)

    return CustomJSONEncoder
