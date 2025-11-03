from ir import ExprType, Expression
from tensor import TensorTable

import torch


class ExpressionCompiler:
    """Compiles expression trees to tensor operations via post-order DFS"""

    EXPR_OPS = {
        ExprType.ADD: torch.add,
        ExprType.SUB: torch.sub,
        ExprType.MUL: torch.mul,
        ExprType.DIV: torch.div,
        ExprType.LT: torch.lt,
        ExprType.GT: torch.gt,
        ExprType.EQ: torch.eq,
        ExprType.LEQ: torch.le,
        ExprType.GEQ: torch.ge,
        ExprType.AND: torch.logical_and,
        ExprType.OR: torch.logical_or,
    }

    @staticmethod
    def compile(expr: Expression, table: TensorTable) -> torch.Tensor:
        """Compile expression tree using post-order DFS traversal"""
        if expr.expr_type == ExprType.COLUMN:
            return table.get_column(expr.value)

        elif expr.expr_type == ExprType.LITERAL:
            device = table.get_column(list(table.columns.keys())[0]).device
            return torch.tensor([expr.value] * len(table), device=device)

        else:
            # Post-order: recursively evaluate children first
            operands = [ExpressionCompiler.compile(child, table) for child in expr.children]

            # Apply corresponding tensor operation
            op_func = ExpressionCompiler.EXPR_OPS[expr.expr_type]
            if len(operands) == 2:
                return op_func(operands[0], operands[1])
            else:
                raise ValueError(f"Unsupported number of operands for {expr.expr_type}")
