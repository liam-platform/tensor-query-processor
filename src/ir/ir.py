from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class OpType(Enum):
    """TQP internal operator types
    Currently it only support a minimal set of core operators."""
    SCAN = "scan"
    FILTER = "filter"
    PROJECT = "project"
    SORT = "sort"
    HASH_JOIN = "hash_join"
    SORT_JOIN = "sort_join"
    GROUP_BY = "group_by"
    LIMIT = "limit"


class ExprType(Enum):
    """Expression types"""
    COLUMN = "column"
    LITERAL = "literal"
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    LT = "lt"
    GT = "gt"
    EQ = "eq"
    LEQ = "leq"
    GEQ = "geq"
    AND = "and"
    OR = "or"
    

@dataclass(kw_only=True)
class Expression:
    expr_type: ExprType
    value: Any = None
    children: Optional[List['Expression']] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


@dataclass
class IRNode:
    """TQP IR node representing a relational operator."""
    op_type: OpType
    children: List['IRNode']
    params: Dict[str, Any]

    def __post_init__(self):
        if self.children is None:
            self.children = []
