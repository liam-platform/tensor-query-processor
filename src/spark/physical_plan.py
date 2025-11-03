from abc import ABC, abstractmethod
from typing import List, Dict


class SparkPhysicalPlan(ABC):
    """Base class for Spark physical plan nodes"""
    @abstractmethod
    def to_dict(self) -> Dict:
        pass


class Project(SparkPhysicalPlan):
    def __init__(self, columns: List[str], child: SparkPhysicalPlan):
        self.columns = columns
        self.child = child

    def to_dict(self):
        return {'type': 'Project', 'columns': self.columns, 'child': self.child.to_dict()}


class Filter(SparkPhysicalPlan):
    def __init__(self, condition: Dict, child: SparkPhysicalPlan):
        self.condition = condition
        self.child = child
    
    def to_dict(self):
        return {'type': 'Filter', 'condition': self.condition, 'child': self.child.to_dict()}


class Sort(SparkPhysicalPlan):
    def __init__(self, key: str, ascending: bool, child: SparkPhysicalPlan):
        self.key = key
        self.ascending = ascending
        self.child = child
    
    def to_dict(self):
        return {'type': 'Sort', 'key': self.key, 'ascending': self.ascending, 'child': self.child.to_dict()}


class HashAggregate(SparkPhysicalPlan):
    def __init__(self, group_cols: List[str], agg_exprs: List[Dict], child: SparkPhysicalPlan):
        self.group_cols = group_cols
        self.agg_exprs = agg_exprs
        self.child = child
    
    def to_dict(self):
        return {'type': 'HashAggregate', 'group_cols': self.group_cols, 
                'agg_exprs': self.agg_exprs, 'child': self.child.to_dict()}


class SortMergeJoin(SparkPhysicalPlan):
    def __init__(self, left_key: str, right_key: str, join_type: str, 
                 left: SparkPhysicalPlan, right: SparkPhysicalPlan):
        self.left_key = left_key
        self.right_key = right_key
        self.join_type = join_type
        self.left = left
        self.right = right
    
    def to_dict(self):
        return {'type': 'SortMergeJoin', 'left_key': self.left_key, 'right_key': self.right_key,
                'join_type': self.join_type, 'left': self.left.to_dict(), 'right': self.right.to_dict()}


class BroadcastHashJoin(SparkPhysicalPlan):
    def __init__(self, left_key: str, right_key: str, join_type: str,
                 left: SparkPhysicalPlan, right: SparkPhysicalPlan):
        self.left_key = left_key
        self.right_key = right_key
        self.join_type = join_type
        self.left = left
        self.right = right
    
    def to_dict(self):
        return {'type': 'BroadcastHashJoin', 'left_key': self.left_key, 'right_key': self.right_key,
                'join_type': self.join_type, 'left': self.left.to_dict(), 'right': self.right.to_dict()}


class FileScan(SparkPhysicalPlan):
    def __init__(self, table_name: str, schema: Dict[str, str]):
        self.table_name = table_name
        self.schema = schema
    
    def to_dict(self):
        return {'type': 'FileScan', 'table': self.table_name, 'schema': self.schema}

