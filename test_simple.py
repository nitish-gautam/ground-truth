"""
Simple test to verify pytest works
"""

def test_simple():
    """Test that basic arithmetic works"""
    assert 2 + 2 == 4

def test_fastapi_imports():
    """Test that FastAPI can be imported"""
    import fastapi
    assert fastapi.FastAPI is not None

def test_pandas_imports():
    """Test that pandas can be imported"""
    import pandas as pd
    import numpy as np

    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    assert len(df) == 3
    assert df['a'].sum() == 6