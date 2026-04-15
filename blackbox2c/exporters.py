"""
Multi-format code exporters for BlackBox2C.

This module provides exporters for different target platforms and languages:
- C++ (classes, templates, modern C++)
- Arduino (sketches, libraries)
- MicroPython (Python modules for microcontrollers)

Each exporter generates platform-specific code from decision trees.
"""

from abc import ABC, abstractmethod
from typing import Union, Optional, List
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np


class BaseExporter(ABC):
    """
    Base class for all code exporters.
    
    Provides common functionality and defines the interface that
    all exporters must implement.
    """
    
    def __init__(
        self,
        function_name: str = "predict",
        use_fixed_point: bool = False,
        precision: int = 8
    ):
        """
        Initialize the exporter.
        
        Parameters
        ----------
        function_name : str
            Name of the prediction function.
        use_fixed_point : bool
            Whether to use fixed-point arithmetic.
        precision : int
            Bit precision (8, 16, or 32).
        """
        self.function_name = function_name
        self.use_fixed_point = use_fixed_point
        self.precision = precision
        self.tree_ = None
        self.feature_names_ = None
        self.class_names_ = None
        self.task_type_ = None
    
    @abstractmethod
    def generate(
        self,
        tree: Union[DecisionTreeClassifier, DecisionTreeRegressor],
        feature_names: List[str],
        class_names: Optional[List[str]] = None
    ) -> str:
        """
        Generate code for the given decision tree.
        
        Parameters
        ----------
        tree : DecisionTreeClassifier or DecisionTreeRegressor
            The decision tree to convert.
        feature_names : list of str
            Names of input features.
        class_names : list of str, optional
            Names of output classes (classification only).
        
        Returns
        -------
        str
            Generated code as a string.
        """
        pass
    
    def _detect_task_type(
        self,
        tree: Union[DecisionTreeClassifier, DecisionTreeRegressor]
    ) -> str:
        """Detect if task is classification or regression."""
        if isinstance(tree, DecisionTreeRegressor):
            return 'regression'
        else:
            return 'classification'
    
    def _get_tree_structure(self, tree):
        """Extract tree structure for code generation."""
        return {
            'n_nodes': tree.tree_.node_count,
            'children_left': tree.tree_.children_left,
            'children_right': tree.tree_.children_right,
            'feature': tree.tree_.feature,
            'threshold': tree.tree_.threshold,
            'value': tree.tree_.value,
            'n_features': tree.n_features_in_
        }


class CppExporter(BaseExporter):
    """
    Export decision trees to modern C++ code.
    
    Generates C++ classes with templates and STL containers.
    Suitable for C++11 and later.
    """
    
    def __init__(
        self,
        function_name: str = "predict",
        class_name: str = "Predictor",
        use_namespace: bool = True,
        namespace: str = "ml",
        use_fixed_point: bool = False,
        precision: int = 8
    ):
        """
        Initialize C++ exporter.
        
        Parameters
        ----------
        function_name : str
            Name of the prediction method.
        class_name : str
            Name of the generated class.
        use_namespace : bool
            Whether to wrap code in a namespace.
        namespace : str
            Namespace name.
        use_fixed_point : bool
            Whether to use fixed-point arithmetic.
        precision : int
            Bit precision (8, 16, or 32).
        """
        super().__init__(function_name, use_fixed_point, precision)
        self.class_name = class_name
        self.use_namespace = use_namespace
        self.namespace = namespace
    
    def generate(
        self,
        tree: Union[DecisionTreeClassifier, DecisionTreeRegressor],
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None
    ) -> str:
        """Generate C++ code."""
        self.tree_ = tree
        n_feat = tree.n_features_in_
        self.feature_names_ = feature_names or [f"feature_{i}" for i in range(n_feat)]
        self.class_names_ = class_names
        self.task_type_ = self._detect_task_type(tree)
        
        code_parts = []
        
        # Header
        code_parts.append(self._generate_header())
        
        # Includes
        code_parts.append(self._generate_includes())
        
        # Namespace opening
        if self.use_namespace:
            code_parts.append(f"\nnamespace {self.namespace} {{\n")
        
        # Class definition
        code_parts.append(self._generate_class())
        
        # Namespace closing
        if self.use_namespace:
            code_parts.append(f"\n}} // namespace {self.namespace}\n")
        
        # Usage example
        code_parts.append(self._generate_usage())
        
        return '\n'.join(code_parts)
    
    def _generate_header(self) -> str:
        """Generate file header comment."""
        return f"""/*
 * Auto-generated C++ code by BlackBox2C
 * 
 * Model Information:
 *   - Input features: {len(self.feature_names_)}
 *   - Task: {self.task_type_.capitalize()}
 *   - Language: C++11
 *   - Fixed-point: {'Yes' if self.use_fixed_point else 'No'}
 * 
 * This code uses modern C++ features for better type safety and performance.
 */"""
    
    def _generate_includes(self) -> str:
        """Generate include statements."""
        includes = [
            "#include <cstdint>",
            "#include <array>",
            "#include <stdexcept>"
        ]
        if self.task_type_ == 'classification':
            includes.append("#include <string>")
        return '\n'.join(includes)
    
    def _generate_class(self) -> str:
        """Generate the predictor class."""
        n_features = len(self.feature_names_)
        
        if self.task_type_ == 'classification':
            return_type = "uint8_t"
            n_classes = len(self.class_names_) if self.class_names_ else 2
        else:
            return_type = "float"
        
        code = f"""
class {self.class_name} {{
public:
    // Feature names
    static constexpr std::array<const char*, {n_features}> FEATURE_NAMES = {{
        {', '.join(f'"{name}"' for name in self.feature_names_)}
    }};
"""
        
        # Add class names for classification
        if self.task_type_ == 'classification':
            code += f"""
    // Class names
    static constexpr std::array<const char*, {n_classes}> CLASS_NAMES = {{
        {', '.join(f'"{name}"' for name in self.class_names_)}
    }};
"""
        
        # Prediction method
        code += f"""
    // Prediction method
    static {return_type} {self.function_name}(const std::array<float, {n_features}>& features) {{
{self._generate_tree_logic(indent=8)}
    }}
"""
        
        # Helper method to get class name (classification only)
        if self.task_type_ == 'classification':
            code += f"""
    // Get class name from prediction
    static const char* get_class_name({return_type} prediction) {{
        if (prediction >= {n_classes}) {{
            throw std::out_of_range("Invalid class index");
        }}
        return CLASS_NAMES[prediction];
    }}
"""
        
        code += "};"
        
        return code
    
    def _generate_tree_logic(self, indent: int = 0) -> str:
        """Generate the decision tree logic."""
        tree_struct = self._get_tree_structure(self.tree_)
        indent_str = ' ' * indent
        
        def recurse(node_id: int, depth: int) -> str:
            node_indent = ' ' * (indent + depth * 4)
            
            # Check if leaf node
            if tree_struct['children_left'][node_id] == tree_struct['children_right'][node_id]:
                # Leaf node
                value = tree_struct['value'][node_id]
                if self.task_type_ == 'classification':
                    class_id = np.argmax(value)
                    return f"{node_indent}return {class_id};"
                else:
                    pred_value = value[0][0]
                    return f"{node_indent}return {pred_value:.6f}f;"
            
            # Internal node
            feature_idx = tree_struct['feature'][node_id]
            threshold = tree_struct['threshold'][node_id]
            feature_name = self.feature_names_[feature_idx]
            
            left_child = tree_struct['children_left'][node_id]
            right_child = tree_struct['children_right'][node_id]
            
            code = f"{node_indent}if (features[{feature_idx}] <= {threshold:.6f}f) {{\n"
            code += recurse(left_child, depth + 1) + "\n"
            code += f"{node_indent}}} else {{\n"
            code += recurse(right_child, depth + 1) + "\n"
            code += f"{node_indent}}}"
            
            return code
        
        return recurse(0, 0)
    
    def _generate_usage(self) -> str:
        """Generate usage example."""
        n_features = len(self.feature_names_)
        
        return f"""
/*
 * Usage Example:
 * 
 *   #include "predictor.hpp"
 *   
 *   int main() {{
 *       std::array<float, {n_features}> features = {{...}};
 *       auto result = {self.namespace}::{self.class_name}::{self.function_name}(features);
 *       
 *       {f'std::cout << {self.namespace}::{self.class_name}::get_class_name(result) << std::endl;' if self.task_type_ == 'classification' else f'std::cout << "Prediction: " << result << std::endl;'}
 *       
 *       return 0;
 *   }}
 */"""


class ArduinoExporter(BaseExporter):
    """
    Export decision trees to Arduino-compatible C++ code.
    
    Generates code optimized for Arduino boards with limited resources.
    Compatible with Arduino IDE and PlatformIO.
    """
    
    def __init__(
        self,
        function_name: str = "predict",
        use_progmem: bool = True,
        use_fixed_point: bool = False,
        precision: int = 8
    ):
        """
        Initialize Arduino exporter.
        
        Parameters
        ----------
        function_name : str
            Name of the prediction function.
        use_progmem : bool
            Whether to store constants in PROGMEM (flash memory).
        use_fixed_point : bool
            Whether to use fixed-point arithmetic.
        precision : int
            Bit precision (8, 16, or 32).
        """
        super().__init__(function_name, use_fixed_point, precision)
        self.use_progmem = use_progmem
    
    def generate(
        self,
        tree: Union[DecisionTreeClassifier, DecisionTreeRegressor],
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None
    ) -> str:
        """Generate Arduino code."""
        self.tree_ = tree
        n_feat = tree.n_features_in_
        self.feature_names_ = feature_names or [f"feature_{i}" for i in range(n_feat)]
        self.class_names_ = class_names
        self.task_type_ = self._detect_task_type(tree)
        
        code_parts = []
        
        # Header
        code_parts.append(self._generate_header())
        
        # Feature names as constants
        code_parts.append(self._generate_feature_constants())
        
        # Class names (if classification)
        if self.task_type_ == 'classification' and self.class_names_:
            code_parts.append(self._generate_class_constants())
        
        # Prediction function
        code_parts.append(self._generate_function())
        
        # Helper functions
        code_parts.append(self._generate_helpers())
        
        # Usage example
        code_parts.append(self._generate_usage())
        
        return '\n'.join(code_parts)
    
    def _generate_header(self) -> str:
        """Generate file header."""
        return f"""/*
 * Auto-generated Arduino code by BlackBox2C
 * 
 * Model Information:
 *   - Input features: {len(self.feature_names_)}
 *   - Task: {self.task_type_.capitalize()}
 *   - PROGMEM: {'Yes' if self.use_progmem else 'No'}
 *   - Fixed-point: {'Yes' if self.use_fixed_point else 'No'}
 * 
 * Compatible with: Arduino Uno, Nano, Mega, ESP8266, ESP32, etc.
 */

#include <Arduino.h>
"""
    
    def _generate_feature_constants(self) -> str:
        """Generate feature name constants."""
        progmem = "PROGMEM " if self.use_progmem else ""
        
        code = f"\n// Feature names\nconst char* const {progmem}FEATURE_NAMES[] = {{\n"
        for name in self.feature_names_:
            code += f'    "{name}",\n'
        code += "};\n"
        
        return code
    
    def _generate_class_constants(self) -> str:
        """Generate class name constants."""
        progmem = "PROGMEM " if self.use_progmem else ""
        
        code = f"\n// Class names\nconst char* const {progmem}CLASS_NAMES[] = {{\n"
        for name in self.class_names_:
            code += f'    "{name}",\n'
        code += "};\n"
        
        return code
    
    def _generate_function(self) -> str:
        """Generate prediction function."""
        n_features = len(self.feature_names_)
        
        if self.task_type_ == 'classification':
            return_type = "uint8_t"
        else:
            return_type = "float"
        
        code = f"""
// Prediction function
{return_type} {self.function_name}(float features[{n_features}]) {{
{self._generate_tree_logic(indent=4)}
}}
"""
        return code
    
    def _generate_tree_logic(self, indent: int = 0) -> str:
        """Generate decision tree logic (reuse from CppExporter)."""
        tree_struct = self._get_tree_structure(self.tree_)
        
        def recurse(node_id: int, depth: int) -> str:
            node_indent = ' ' * (indent + depth * 4)
            
            if tree_struct['children_left'][node_id] == tree_struct['children_right'][node_id]:
                value = tree_struct['value'][node_id]
                if self.task_type_ == 'classification':
                    class_id = np.argmax(value)
                    return f"{node_indent}return {class_id};"
                else:
                    pred_value = value[0][0]
                    return f"{node_indent}return {pred_value:.6f}f;"
            
            feature_idx = tree_struct['feature'][node_id]
            threshold = tree_struct['threshold'][node_id]
            left_child = tree_struct['children_left'][node_id]
            right_child = tree_struct['children_right'][node_id]
            
            code = f"{node_indent}if (features[{feature_idx}] <= {threshold:.6f}f) {{\n"
            code += recurse(left_child, depth + 1) + "\n"
            code += f"{node_indent}}} else {{\n"
            code += recurse(right_child, depth + 1) + "\n"
            code += f"{node_indent}}}"
            
            return code
        
        return recurse(0, 0)
    
    def _generate_helpers(self) -> str:
        """Generate helper functions."""
        if self.task_type_ == 'classification':
            return """
// Get class name from prediction
const char* get_class_name(uint8_t prediction) {
    return CLASS_NAMES[prediction];
}
"""
        return ""
    
    def _generate_usage(self) -> str:
        """Generate usage example."""
        n_features = len(self.feature_names_)
        
        return f"""
/*
 * Arduino Sketch Example:
 * 
 * void setup() {{
 *     Serial.begin(9600);
 * }}
 * 
 * void loop() {{
 *     float features[{n_features}];
 *     
 *     // Read sensor values
 *     features[0] = analogRead(A0) / 1023.0;
 *     features[1] = analogRead(A1) / 1023.0;
 *     // ... fill other features
 *     
 *     // Make prediction
 *     {'uint8_t result = ' + self.function_name + '(features);' if self.task_type_ == 'classification' else 'float result = ' + self.function_name + '(features);'}
 *     
 *     // Print result
 *     Serial.print("Prediction: ");
 *     {f'Serial.println(get_class_name(result));' if self.task_type_ == 'classification' else 'Serial.println(result);'}
 *     
 *     delay(1000);
 * }}
 */"""


class MicroPythonExporter(BaseExporter):
    """
    Export decision trees to MicroPython code.
    
    Generates Python modules optimized for microcontrollers running MicroPython.
    Compatible with ESP32, ESP8266, Raspberry Pi Pico, etc.
    """
    
    def __init__(
        self,
        function_name: str = "predict",
        class_name: str = "Predictor",
        use_const: bool = True
    ):
        """
        Initialize MicroPython exporter.
        
        Parameters
        ----------
        function_name : str
            Name of the prediction function.
        class_name : str
            Name of the predictor class.
        use_const : bool
            Whether to use const() for memory optimization.
        """
        super().__init__(function_name, False, 8)  # MicroPython uses float
        self.class_name = class_name
        self.use_const = use_const
    
    def generate(
        self,
        tree: Union[DecisionTreeClassifier, DecisionTreeRegressor],
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None
    ) -> str:
        """Generate MicroPython code."""
        self.tree_ = tree
        n_feat = tree.n_features_in_
        self.feature_names_ = feature_names or [f"feature_{i}" for i in range(n_feat)]
        self.class_names_ = class_names
        self.task_type_ = self._detect_task_type(tree)
        
        code_parts = []
        
        # Header
        code_parts.append(self._generate_header())
        
        # Imports
        code_parts.append(self._generate_imports())
        
        # Class definition
        code_parts.append(self._generate_class())
        
        # Usage example
        code_parts.append(self._generate_usage())
        
        return '\n'.join(code_parts)
    
    def _generate_header(self) -> str:
        """Generate file header."""
        return f'''"""
Auto-generated MicroPython code by BlackBox2C

Model Information:
  - Input features: {len(self.feature_names_)}
  - Task: {self.task_type_.capitalize()}
  - Memory optimization: {'Yes' if self.use_const else 'No'}

Compatible with: ESP32, ESP8266, Raspberry Pi Pico, PyBoard, etc.
"""'''
    
    def _generate_imports(self) -> str:
        """Generate import statements."""
        imports = []
        if self.use_const:
            imports.append("from micropython import const")
        return '\n'.join(imports) if imports else ""
    
    def _generate_class(self) -> str:
        """Generate predictor class."""
        n_features = len(self.feature_names_)
        
        code = f"""

class {self.class_name}:
    \"\"\"Decision tree predictor for {self.task_type_}.\"\"\"
    
    # Feature names
    FEATURE_NAMES = {self.feature_names_}
"""
        
        # Add class names for classification
        if self.task_type_ == 'classification' and self.class_names_:
            code += f"""    
    # Class names
    CLASS_NAMES = {self.class_names_}
"""
        
        # Prediction method
        code += f"""
    @staticmethod
    def {self.function_name}(features):
        \"\"\"
        Make a prediction.
        
        Args:
            features: List or tuple of {n_features} feature values
        
        Returns:
            {'Class index (int)' if self.task_type_ == 'classification' else 'Predicted value (float)'}
        \"\"\"
        if len(features) != {n_features}:
            raise ValueError(f"Expected {n_features} features, got {{len(features)}}")
        
{self._generate_tree_logic(indent=8)}
"""
        
        # Helper method for classification
        if self.task_type_ == 'classification':
            code += """
    @staticmethod
    def get_class_name(prediction):
        \"\"\"Get class name from prediction index.\"\"\"
        return Predictor.CLASS_NAMES[prediction]
"""
        
        return code
    
    def _generate_tree_logic(self, indent: int = 0) -> str:
        """Generate decision tree logic in Python."""
        tree_struct = self._get_tree_structure(self.tree_)
        
        def recurse(node_id: int, depth: int) -> str:
            node_indent = ' ' * (indent + depth * 4)
            
            if tree_struct['children_left'][node_id] == tree_struct['children_right'][node_id]:
                value = tree_struct['value'][node_id]
                if self.task_type_ == 'classification':
                    class_id = int(np.argmax(value))
                    return f"{node_indent}return {class_id}"
                else:
                    pred_value = float(value[0][0])
                    return f"{node_indent}return {pred_value:.6f}"
            
            feature_idx = tree_struct['feature'][node_id]
            threshold = tree_struct['threshold'][node_id]
            left_child = tree_struct['children_left'][node_id]
            right_child = tree_struct['children_right'][node_id]
            
            code = f"{node_indent}if features[{feature_idx}] <= {threshold:.6f}:\n"
            code += recurse(left_child, depth + 1) + "\n"
            code += f"{node_indent}else:\n"
            code += recurse(right_child, depth + 1)
            
            return code
        
        return recurse(0, 0)
    
    def _generate_usage(self) -> str:
        """Generate usage example."""
        n_features = len(self.feature_names_)
        
        return f'''

"""
Usage Example:

    from predictor import {self.class_name}
    
    # Prepare features
    features = [...]  # {n_features} values
    
    # Make prediction
    result = {self.class_name}.{self.function_name}(features)
    
    # Print result
    {f'print("Predicted class:", {self.class_name}.get_class_name(result))' if self.task_type_ == 'classification' else 'print("Prediction:", result)'}
    
    # Example with sensor readings
    from machine import ADC
    
    adc0 = ADC(0)
    adc1 = ADC(1)
    
    while True:
        features = [
            adc0.read() / 1023.0,
            adc1.read() / 1023.0,
            # ... other features
        ]
        
        result = {self.class_name}.{self.function_name}(features)
        print("Prediction:", result)
        
        time.sleep(1)
"""'''


# Factory function for easy exporter creation
def create_exporter(
    format: str,
    **kwargs
) -> BaseExporter:
    """
    Create an exporter for the specified format.
    
    Parameters
    ----------
    format : str
        Export format: 'cpp', 'arduino', or 'micropython'
    **kwargs
        Additional arguments passed to the exporter constructor
    
    Returns
    -------
    BaseExporter
        An instance of the appropriate exporter
    
    Raises
    ------
    ValueError
        If format is not supported
    
    Examples
    --------
    >>> exporter = create_exporter('cpp', class_name='MyPredictor')
    >>> code = exporter.generate(tree, feature_names)
    """
    exporters = {
        'cpp': CppExporter,
        'c++': CppExporter,
        'arduino': ArduinoExporter,
        'micropython': MicroPythonExporter,
        'python': MicroPythonExporter
    }
    
    format_lower = format.lower()
    if format_lower not in exporters:
        raise ValueError(
            f"Unsupported format: {format}. "
            f"Supported formats: {', '.join(exporters.keys())}"
        )
    
    return exporters[format_lower](**kwargs)
