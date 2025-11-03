from typing import Dict, List, Tuple

from tensor import TensorTable


class TQPExecutor:
    """PyTorch executor object that manages tensor program execution"""

    def __init__(self, plan: List[Tuple[str, callable, Dict]]):
        self.plan = plan

    def execute(self, tables: Dict[str, TensorTable]) -> TensorTable:
        """Execute the operator plan in topological order

        Manages:
        - Calling tensor programs in order
        - Wiring output tensors to next program
        - Garbage collection of unused tensors
        """
        result = None
        current_left = None
        current_right = None

        for op_name, op_func, params in self.plan:
            print("op_func = ", op_func)
            if op_name == 'scan':
                # Initialize result with the scanned table
                result = tables[params['table']]

            elif op_name == 'filter':
                result = op_func(result, params['condition'])
                print("result: ", result)

            elif op_name == 'project':
                result = op_func(result, params['columns'])

            elif op_name == 'sort':
                result = op_func(result, params['key'], params.get('ascending', True))

            elif op_name == 'sort_join':
                # For joins, need to track left and right inputs
                if current_left is None:
                    current_left = result
                    result = None # Clear result after assigning to current_left
                elif current_right is None:
                    current_right = result
                    result = op_func(current_left, current_right,
                                   params['left_key'], params['right_key'])
                    current_left = None
                    current_right = None

            elif op_name == 'hash_join':
                if current_left is None:
                    current_left = result
                    result = None # Clear result after assigning to current_left
                elif current_right is None:
                    current_right = result
                    result = op_func(current_left, current_right,
                                   params['left_key'], params['right_key'])
                    current_left = None
                    current_right = None

            elif op_name == 'group_by':
                agg_expr = params['agg_exprs'][0]  # Simplified - take first agg
                result = op_func(result, params['group_cols'],
                               agg_expr['column'], agg_expr['function'])

            # Garbage collection happens automatically via Python reference counting
            # Unused tensors are freed when overwritten

        return result

    def to_torchscript(self):
        """Compile executor to TorchScript format"""
        # Can be compiled to TorchScript for deployment
        pass

    def to_onnx(self):
        """Export executor to ONNX format"""
        # Can be exported to ONNX for cross-platform execution
        pass
