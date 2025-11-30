# Building an Intermediate Representation (IR) for Spark Physical Plans

## And Executing Queries on Another Engine

This document describes how to design and implement an Intermediate Representation (IR) that you can derive from a Spark Physical Plan and use to execute queries on another execution engine (custom engine, vector DB, in-house compute layer, etc.).

The goal is:

- Parse Spark’s physical plan → Convert to a stable IR → Execute on another engine
- Without copying Spark internals, but keeping enough semantic

```md
Spark Logical Plan
        ↓
Catalyst Optimizer
        ↓
Spark Physical Plan
        ↓
─────────────── Your Translation Layer ────────────────
        ↓
      IR (Intermediate Representation)
        ↓
  Your Engine's Physical Operators
        ↓
      Execution
```

## Requirements for the IR
A good IR should be:

✔ Engine-neutral

No Spark-specific class names: avoid HashAggregateExec, ProjectExec, FilterExec.
Instead use generic operators:

- Scan
- Filter
- Project
- Aggregate
- Join
- Sort
- Limit
- Exchange (optional)
- Window (optional)
- UDF or Function

✔ Graph-structured

- Operators form a directed acyclic graph (DAG).
- Children = inputs; parent = consumer.

✔ Expression trees supported

- Every transformation and predicate must be representable as an expression tree:
  - BinaryOp(">", Column("amount"), Literal(100))
  - Function("substring", [Column("name"), Literal(1), Literal(3)])
  - CaseWhen([...])

✔ Explicit schema representation

IR must track:

- attribute name
- type
- nullable flag
- field metadata (optional)
- deterministic flag (optional)

✔ Removable Spark-specific elements

Spark uses:

- column IDs (colName#23)
- codegen markers (*(2))
- AQE nodes (AdaptiveSparkPlan)
- partitioning info (hashpartitioning(col))

Our IR should normalize or drop these.

## IR Node Definition
### Core Operators
```scss
IRNode
 ├── IRScan(table, schema, filters, columns)
 ├── IRFilter(condition)
 ├── IRProject(expressions)
 ├── IRAggregate(grouping_keys, aggregate_exprs)
 ├── IRJoin(type, condition)
 ├── IRSort(ordering)
 ├── IRLimit(n)
 ├── IRWindow(window_exprs)
 ├── IRExchange(partitioning)   # optional
 └── IRUDF(name, args)
```
We can define them using JSON, protobuf, FlatBuffers, or internal classes.

## Pipeline to Convert Spark Physical Plan → IR
*Step 1: Retrieve Spark Physical Plan*

*Step 2: Walk the Spark Execution Plan*

Spark plan is a tree of SparkPlan nodes.
Use pattern matching:
```scala
plan match {
  case ProjectExec(projectList, child) => ...
  case FilterExec(condition, child) => ...
  case HashAggregateExec(...) => ...
  case SortMergeJoinExec(...) => ...
  ...
}
```

*Step 3: Convert each Spark operator to IR operator*

Example:

Spark
```sh
Filter (amount#22 > 100)
```
Convert to IR
```less
IRFilter(
    condition = BinaryOp(">", Column("amount"), Literal(100))
)
```

Spark
```perl
HashAggregate(keys=[category#10], functions=[sum(amount#22)])
```
Convert to IR
```less
IRAggregate(
    grouping_keys = [Column("category")],
    aggregate_exprs = [AggregateFunction("sum", Column("amount"))]
)
```

*Step 4: Convert Spark expressions to your expression IR*
Spark has expression classes like:
```sh
Add

Multiply

GreaterThan

EqualTo

Alias

Literal

AttributeReference
```

Match and convert them to our representation.


## A Smart Shortcut: Use Substrait Instead of Inventing IR
Substrait is a standard IR made exactly for this purpose. Many engines already support Substrait. 

Flow becomes:
```sh
Spark Physical Plan → Substrait Plan → Engine Plan → Execution
```
