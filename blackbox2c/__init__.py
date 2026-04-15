"""
BlackBox2C - Convert ML models to C code for embedded systems.
"""

from .converter import Converter
from .config import ConversionConfig
from .analysis import FeatureSensitivityAnalyzer, SensitivityResults
from .exporters import (
    BaseExporter,
    CppExporter,
    ArduinoExporter,
    MicroPythonExporter,
    create_exporter
)

__version__ = '0.1.0'
__all__ = [
    'Converter',
    'ConversionConfig',
    'convert',
    'FeatureSensitivityAnalyzer',
    'SensitivityResults',
    'BaseExporter',
    'CppExporter',
    'ArduinoExporter',
    'MicroPythonExporter',
    'create_exporter'
]


def convert(
    model,
    X_train,
    feature_names=None,
    class_names=None,
    X_test=None,
    target='c',
    config=None,
    **config_kwargs
):
    """
    Convenience function to convert a model to embedded code.
    
    Parameters
    ----------
    model : BaseEstimator
        Trained scikit-learn model.
    X_train : np.ndarray
        Training data.
    feature_names : list, optional
        Names of features.
    class_names : list, optional
        Names of classes.
    X_test : np.ndarray, optional
        Test data for fidelity evaluation.
    target : str, default='c'
        Output format. One of 'c', 'cpp', 'arduino', 'micropython'.
    config : ConversionConfig, optional
        Configuration object. If provided, config_kwargs are ignored.
    **config_kwargs
        Additional configuration parameters passed to ConversionConfig.
        Only used if config is None.
    
    Returns
    -------
    str
        Generated code in the requested target format.
    
    Examples
    --------
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from blackbox2c import convert
    >>> model = DecisionTreeClassifier()
    >>> model.fit(X_train, y_train)
    >>> c_code = convert(model, X_train, max_depth=5)
    
    >>> # Export to Arduino
    >>> arduino_code = convert(model, X_train, target='arduino')
    
    >>> # With feature selection
    >>> c_code = convert(model, X_train, feature_threshold=3)
    
    >>> # With custom config
    >>> from blackbox2c import ConversionConfig
    >>> config = ConversionConfig(max_depth=7, optimize_rules='high')
    >>> c_code = convert(model, X_train, config=config)
    """
    if config is None:
        config = ConversionConfig(**config_kwargs)
    converter = Converter(config)
    return converter.convert(model, X_train, feature_names, class_names, X_test, target=target)
