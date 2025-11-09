from ir import IRNode, OpType
from relational_operator import RelationalOperators

from typing import Callable, Dict, List, Tuple


class TQPPlanner:
    """Transforms IR graph to operator plan (tensor programs)"""

    def __init__(self):
        # Dictionary mapping operators to tensor program implementations
        self.operator_dict = {
            OpType.FILTER: RelationalOperators.filter,
            OpType.PROJECT: RelationalOperators.project,
            OpType.SORT: RelationalOperators.sort,
            OpType.SORT_JOIN: RelationalOperators.sort_merge_join,
            OpType.HASH_JOIN: RelationalOperators.hash_join,
            OpType.GROUP_BY: RelationalOperators.group_by,
            OpType.SCAN: RelationalOperators.scan,
        }

    def plan(self, ir: IRNode) -> List[Tuple[str, Callable, Dict]]:
        """Convert IR graph to operator plan"""
        plan = []
        self._build_plan(ir, plan)
        return plan

    def _build_plan(self, node: IRNode, plan: List):
        """Build plan via DFS post-order traversal"""
        # Process children first
        for child in node.children:
            self._build_plan(child, plan)

        # Fetch corresponding tensor program from dictionary
        if node.op_type in self.operator_dict:
            plan.append((
                node.op_type.value,
                self.operator_dict[node.op_type],
                node.params
            ))
