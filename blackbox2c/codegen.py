"""
C code generation from decision trees.

This module converts decision tree structures into optimized C code
with if-else statements. Supports both classification and regression tasks.
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from typing import List, Optional, Tuple, Union


class CCodeGenerator:
    """
    Generate optimized C code from decision trees.
    
    Supports both classification and regression tasks. The task type is
    automatically detected from the tree structure.
    """
    
    def __init__(
        self,
        function_name: str = 'predict',
        use_fixed_point: bool = False,
        precision: int = 8,
        optimize_rules: str = 'medium',
        task_type: Optional[str] = None
    ):
        """
        Initialize the C code generator.
        
        Parameters
        ----------
        function_name : str
            Name of the generated function.
        use_fixed_point : bool
            Use fixed-point arithmetic instead of float.
        precision : int
            Bit precision (8, 16, or 32).
        optimize_rules : str
            Optimization level: 'low', 'medium', or 'high'.
        task_type : str, optional
            Task type: 'classification' or 'regression'.
            If None, auto-detected from tree.
        """
        self.function_name = function_name
        self.use_fixed_point = use_fixed_point
        self.precision = precision
        self.optimize_rules = optimize_rules
        self.task_type = task_type
        self.indent_size = 4
    
    def generate(
        self,
        tree: Union[DecisionTreeClassifier, DecisionTreeRegressor],
        feature_names: List[str],
        class_names: Optional[List[str]] = None
    ) -> str:
        """
        Generate C code from a decision tree.
        
        Parameters
        ----------
        tree : DecisionTreeClassifier or DecisionTreeRegressor
            The trained decision tree.
        feature_names : list
            Names of input features.
        class_names : list, optional
            Names of output classes (only for classification).
        
        Returns
        -------
        str
            Complete C code as a string.
        """
        # Auto-detect task type if not specified
        if self.task_type is None:
            if isinstance(tree, DecisionTreeRegressor):
                self.task_type = 'regression'
            elif hasattr(tree, 'n_classes_'):
                self.task_type = 'classification'
            else:
                # Try to infer from tree structure
                if hasattr(tree.tree_, 'n_classes') and tree.tree_.n_classes[0] > 1:
                    self.task_type = 'classification'
                else:
                    self.task_type = 'regression'
        
        n_features = len(feature_names)
        
        # Handle classification vs regression
        if self.task_type == 'classification':
            n_classes = tree.n_classes_ if hasattr(tree, 'n_classes_') else 2
            if class_names is None:
                class_names = [f"CLASS_{i}" for i in range(n_classes)]
        else:
            n_classes = None
            class_names = None
        
        # Generate header
        code_parts = []
        code_parts.append(self._generate_header(n_features, n_classes))
        
        if self.task_type == 'classification':
            code_parts.append(self._generate_defines(class_names))
        
        code_parts.append(self._generate_function_signature(n_features))
        
        # Generate decision tree logic
        tree_code = self._generate_tree_logic(
            tree.tree_,
            feature_names,
            node_id=0,
            depth=1
        )
        code_parts.append(tree_code)
        
        # Close function
        code_parts.append("}\n")
        
        # Add helper comment
        if self.task_type == 'classification':
            code_parts.append(self._generate_usage_comment(feature_names, class_names))
        else:
            code_parts.append(self._generate_usage_comment_regression(feature_names))
        
        return "\n".join(code_parts)
    
    def _generate_header(self, n_features: int, n_classes: Optional[int]) -> str:
        """Generate file header with comments."""
        if self.task_type == 'classification':
            task_info = f"*   - Output classes: {n_classes}\n"
        else:
            task_info = "*   - Task: Regression\n"
        
        return f"""/*
 * Auto-generated C code by BlackBox2C
 *
 * Model Information:
 *   - Input features: {n_features}
 *   {task_info.strip()}
 *   - Precision: {self.precision}-bit
 *   - Fixed-point: {'Yes' if self.use_fixed_point else 'No'}
 *
 * This code is optimized for embedded systems with limited resources.
 */

#include <stdint.h>
"""
    
    def _generate_defines(self, class_names: List[str]) -> str:
        """Generate class label defines."""
        defines = ["\n/* Class labels */"]
        for i, name in enumerate(class_names):
            defines.append(f"#define {name} {i}")
        defines.append("")
        return "\n".join(defines)
    
    def _generate_function_signature(self, n_features: int) -> str:
        """Generate function signature."""
        if self.use_fixed_point:
            if self.precision == 8:
                dtype = "int8_t"
            elif self.precision == 16:
                dtype = "int16_t"
            else:
                dtype = "int32_t"
        else:
            dtype = "float"
        
        # Return type depends on task
        if self.task_type == 'classification':
            return_type = "uint8_t"
        else:
            return_type = "float"
        
        return f"""
/* Prediction function */
{return_type} {self.function_name}({dtype} features[{n_features}]) {{"""
    
    def _generate_tree_logic(
        self,
        tree,
        feature_names: List[str],
        node_id: int,
        depth: int
    ) -> str:
        """
        Recursively generate if-else logic for the tree.
        
        Parameters
        ----------
        tree : Tree object
            The tree structure from sklearn.
        feature_names : list
            Feature names.
        node_id : int
            Current node ID.
        depth : int
            Current depth (for indentation).
        
        Returns
        -------
        str
            C code for this subtree.
        """
        indent = " " * (self.indent_size * depth)
        
        # Check if leaf node
        if tree.feature[node_id] == -2:  # Leaf node
            if self.task_type == 'classification':
                # Get the predicted class
                class_counts = tree.value[node_id][0]
                predicted_class = np.argmax(class_counts)
                return f"{indent}return {predicted_class};"
            else:
                # Get the predicted value (regression)
                predicted_value = tree.value[node_id][0][0]
                return f"{indent}return {predicted_value:.6f}f;"
        
        # Internal node - generate condition
        feature_idx = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        feature_name = feature_names[feature_idx]
        
        # Format threshold based on precision
        if self.use_fixed_point:
            threshold_str = str(int(threshold * (2 ** (self.precision - 1))))
        else:
            threshold_str = f"{threshold:.6f}f"
        
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]
        
        # Generate condition
        code = f"{indent}if ({feature_name} <= {threshold_str}) {{\n"
        
        # Recursively generate left subtree
        code += self._generate_tree_logic(tree, feature_names, left_child, depth + 1)
        code += f"\n{indent}}} else {{\n"
        
        # Recursively generate right subtree
        code += self._generate_tree_logic(tree, feature_names, right_child, depth + 1)
        code += f"\n{indent}}}"
        
        return code
    
    def _generate_usage_comment(
        self,
        feature_names: List[str],
        class_names: List[str]
    ) -> str:
        """Generate usage example comment for classification."""
        features_str = ", ".join(feature_names)
        classes_str = ", ".join(class_names)
        
        return f"""
/*
 * Usage Example:
 * 
 *   float input[{len(feature_names)}] = {{...}};  // Your feature values
 *   uint8_t result = {self.function_name}(input);
 * 
 * Input features: {features_str}
 * Output classes: {classes_str}
 */
"""
    
    def _generate_usage_comment_regression(self, feature_names: List[str]) -> str:
        """Generate usage example comment for regression."""
        features_str = ", ".join(feature_names)
        
        return f"""
/*
 * Usage Example:
 * 
 *   float input[{len(feature_names)}] = {{...}};  // Your feature values
 *   float result = {self.function_name}(input);
 * 
 * Input features: {features_str}
 * Output: Continuous value (float)
 */
"""
    
    def estimate_code_size(self, tree: DecisionTreeClassifier) -> dict:
        """
        Estimate the size of generated code.
        
        Returns
        -------
        dict
            Dictionary with size estimates in bytes.
        """
        n_nodes = tree.tree_.node_count
        n_leaves = np.sum(tree.tree_.feature == -2)
        n_internal = n_nodes - n_leaves
        
        # Rough estimates
        bytes_per_condition = 12  # if statement
        bytes_per_return = 4      # return statement
        overhead = 50             # function overhead
        
        estimated_flash = (
            overhead +
            n_internal * bytes_per_condition +
            n_leaves * bytes_per_return
        )
        
        # RAM usage during execution (minimal - just stack)
        estimated_ram = 32  # Stack frame
        
        return {
            'flash_bytes': estimated_flash,
            'ram_bytes': estimated_ram,
            'n_nodes': n_nodes,
            'n_conditions': n_internal,
            'n_leaves': n_leaves
        }
