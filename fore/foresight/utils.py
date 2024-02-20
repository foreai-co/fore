"""Utils for the foresight package."""
from typing import Any, Dict


def convert_to_pandas_dataframe(data: Dict[str, Any]) -> "pandas.DataFrame":
    """Converts a dictionary to a pandas DataFrame."""
    import pandas as pd  # pylint: disable=import-outside-toplevel

    return pd.DataFrame(data)
