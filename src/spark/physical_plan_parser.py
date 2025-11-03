from physical_plan import SparkPhysicalPlan

from ir import IRNode, OpType, Expression, ExprType
from typing import Dict


class SparkPhysicalPlanParser:
    """Converts Spark Catalyst physical query plan to TQP IR graph

    Uses DFS post-order traversal
    """

    def parse(self, spark_plan: SparkPhysicalPlan) -> IRNode:
        """Convert Spark physical plan to TQP IR using DFS post-order traversal"""
        plan_dict = spark_plan.to_dict()
        return self._parse_node(plan_dict)

    def _parse_node(self, node: Dict) -> IRNode:
        """Recursively parse Spark plan node"""
        node_type = node['type']

        if node_type == 'FileScan':
            return IRNode(
                op_type=OpType.SCAN,
                children=[],
                params={'table': node['table'], 'schema': node['schema']}
            )

        elif node_type == 'Filter':
            child_ir = self._parse_node(node['child'])
            condition = self._parse_condition(node['condition'])
            return IRNode(
                op_type=OpType.FILTER,
                children=[child_ir],
                params={'condition': condition}
            )

        elif node_type == 'Project':
            child_ir = self._parse_node(node['child'])
            return IRNode(
                op_type=OpType.PROJECT,
                children=[child_ir],
                params={'columns': node['columns']}
            )

        elif node_type == 'Sort':
            child_ir = self._parse_node(node['child'])
            return IRNode(
                op_type=OpType.SORT,
                children=[child_ir],
                params={'key': node['key'], 'ascending': node['ascending']}
            )

        elif node_type == 'SortMergeJoin':
            left_ir = self._parse_node(node['left'])
            right_ir = self._parse_node(node['right'])
            return IRNode(
                op_type=OpType.SORT_JOIN,
                children=[left_ir, right_ir],
                params={
                    'left_key': node['left_key'],
                    'right_key': node['right_key'],
                    'join_type': node['join_type']
                }
            )

        elif node_type == 'BroadcastHashJoin':
            left_ir = self._parse_node(node['left'])
            right_ir = self._parse_node(node['right'])
            return IRNode(
                op_type=OpType.HASH_JOIN,
                children=[left_ir, right_ir],
                params={
                    'left_key': node['left_key'],
                    'right_key': node['right_key'],
                    'join_type': node['join_type']
                }
            )

        elif node_type == 'HashAggregate':
            child_ir = self._parse_node(node['child'])
            return IRNode(
                op_type=OpType.GROUP_BY,
                children=[child_ir],
                params={
                    'group_cols': node['group_cols'],
                    'agg_exprs': node['agg_exprs']
                }
            )

        else:
            raise ValueError(f"Unsupported Spark operator: {node_type}")

    def _parse_condition(self, cond: Dict) -> Expression:
        """Parse filter condition to Expression tree"""
        op = cond['op']

        if op == 'column':
            return Expression(expr_type=ExprType.COLUMN, value=cond['name'])
        elif op == 'literal':
            return Expression(expr_type=ExprType.LITERAL, value=cond['value'])
        elif op == '<':
            return Expression(expr_type=ExprType.LT, children=[
                self._parse_condition(cond['left']),
                self._parse_condition(cond['right'])
            ])
        elif op == '>':
            return Expression(expr_type=ExprType.GT, children=[
                self._parse_condition(cond['left']),
                self._parse_condition(cond['right'])
            ])
        elif op == '=':
            return Expression(expr_type=ExprType.EQ, children=[
                self._parse_condition(cond['left']),
                self._parse_condition(cond['right'])
            ])
        else:
            raise ValueError(f"Unsupported condition operator: {op}")
