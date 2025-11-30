# DEVELOPMENT

This document describes the architecture, implementation design and known problems of the current project, with emphasis on the TQPExecutor implementation and recommended improvements.

---

## Table of contents

- Project overview
- Directory layout
- Core concepts and data model
- Plan representation (current)
- TQPExecutor: current execution semantics (detailed)
- Problems and failure modes (detailed)
- Recommended design and API contract
- Migration plan and minimal replacement
- Testing and validation recommendations
- Extension points

---

## Project overview

This repository implements a simple tensor-query pipeline (TQP) stack:

- planner: produces a logical/physical plan for relational operations
- compiler: lowers plans to executable operators / tensor programs
- executor: runs compiled operators on TensorTable data
- expr/ir: expression and IR utilities
- relational_operator: higher-level operator definitions
- tensor: table-like tensor-backed datasets

The executor is the runtime that wires operators together and executes a plan against provided in-memory TensorTable inputs.

---

## Directory layout (important files)

- src/planner/tqp_planner.py — planner that produces plan tuples
- src/compiler/tqp_compiler.py — compiler that maps plan to operators
- src/executor/tqp_executor.py — current executor implementation (focus of this doc)
- src/relational_operator/relational_operator.py — op implementations / helpers
- src/tensor/tensor_table.py — TensorTable type and accessors
- src/tests/test_functionality.py — functional tests

---

## Core concepts and data model

- TensorTable: the project's primary data structure for tabular tensor-backed data. Operators accept and return TensorTable.
- Operator: a callable that consumes one or more TensorTable instances and returns a TensorTable.
- Plan: a linear sequence of steps the executor walks. Each step is currently represented as a tuple (op_name, op_func, params).

Operator expectations (implicit in current code):

- scan: params['table'] -> reads from input map
- filter: op_func(table, condition)
- project: op_func(table, columns)
- sort/sort_join/hash_join: op_func variants, sometimes binary
- group_by: op_func(table, group_cols, agg_column, agg_fn)

These signatures are not consistently enforced.

---

## (Legacy) Plan representation

Type: List[Tuple[str, callable, Dict]]

Example element:

- ("scan", scan_func, {"table": "users"})
- ("filter", filter_func, {"condition": "<expr>"})

Plan is treated as a topological list but the executor uses position and a transient `result` variable and two state slots (`current_left`, `current_right`) to manage inputs for binary joins.

## (New) Plan representation

New explicit nodes:

```
{
    "op": "filter",
    "op_func": callable,
    "inputs": ["scan_users"],
    "output": "filtered_users",
    "params": {...}
}
```

> New style is preferred: it uses a named-environment model (env) and a stable operator signature: Callable[[List[TensorTable], Dict], TensorTable]. Legacy style is adapted to the new env model where possible.

---

## TQPExecutor: current execution semantics (detailed)

Implementation summary:

- Maintains three local slots: `result`, `current_left`, `current_right`.
- Iterates over plan entries in order.
- `scan` assigns `result = tables[params['table']]`.
- Unary ops (filter / project / sort / group_by) call op_func(result, ...).
- Binary joins: uses `current_left` and `current_right` as a two-step state machine:
  - First join-related op encountered assigns `current_left = result` and clears `result`.
  - On the next join-related op, assigns `current_right = result` and executes join with (current_left, current_right).
- After join, clears the left/right state and stores join result in `result`.
- No explicit resource lifecycle management beyond trusting Python GC.

---

## Problems and failure modes (technical)

1. Implicit single-stream model

   - The executor assumes a single flowing `result`. This breaks for graphs where an operator consumes named intermediate results that are not its immediate predecessor.
   - Multi-source operators (joins with non-consecutive inputs) are fragile.

2. Fragile join state machine

   - `current_left` and `current_right` rely on exact ordering of plan entries and presence of `result` when expected.
   - If scans or other ops interleave differently, the assignment semantics break.
   - No explicit input identifiers: the executor cannot know which previous plan entry corresponds to left vs right.

3. No named intermediate results / overwrites

   - Scans and operator outputs overwrite the single `result` slot. If the plan needs to reference two previous outputs that were produced earlier (non-last), they are lost.
   - Multiple scans overwrite without storing earlier scanned tables under names.

4. Operator signature ambiguity

   - No formal operator API. Executor calls operators with positional parameters assumed by name. Some ops may expect different shapes or param passing style.
   - group_by uses only the first aggregation and assumes agg_expr structure with keys 'column' and 'function'.

5. Poor validation and error handling

   - Missing parameter keys (e.g., missing 'table' on scan) will raise KeyError deep inside execution rather than a clear plan validation error.
   - Empty plan returns `None` silently.
   - Invalid op_name leads to no-op (silent skip), not an explicit error.

6. Hidden side effects and non-determinism

   - Logic depends on Python GC for tensor cleanup with no clear ownership model (view vs copy).
   - Prints are left in code which pollutes logs.

7. Limited expressivity

   - Cannot express multi-input DAGs, fanout (one result used by multiple downstream ops), or reuse intermediate results.

8. Maintainability issues
   - Duplicate code for sort_join/hash_join state machine.
   - Implicit assumptions are scattered and not documented.

---

## Recommended design and API contract

1. Explicit plan schema (prefer JSON-serializable dicts)

   - Each plan node: {
     "op": "filter",
     "op_func": <callable or op_id>,
     "inputs": ["t1"], // list of names
     "output": "t2", // name for the result
     "params": {...} // op-specific params
     }

2. Maintain environment map

   - env: Dict[str, TensorTable] seeded with input tables.
   - Each plan node resolves `inputs` by name from env, calls op_func with explicit args, stores under `output`.

3. Operator signature

   - Standard op signature: Callable[[List[TensorTable], Dict[str, Any]], TensorTable]
     - Always receive inputs as list and params as dict.
     - Jobs: op must not mutate inputs in place (or must document mutability).
     - Provide thin adapter layer for existing operators that use different signatures.

4. Validation phase

   - Validate arity, required params, input presence prior to execution.
   - Provide useful exceptions with node identifiers.

5. Joins and multi-input ops

   - Express as nodes with inputs ["left_name", "right_name"] and op that receives both tables in order.

6. Deterministic resource handling

   - Document which ops produce views vs copies.
   - If necessary, add explicit `drop` nodes or `retain` flags.

7. Remove print statements; use logging at DEBUG level.

---

## Minimal replacement execute pseudocode

The executor `execute` should:

- Build env = dict(tables) // seed with input tables by name
- For each node in plan:
  - Validate node.inputs exists in env
  - Gather args = [env[name] for name in node.inputs]
  - Call result = op_func(args, node.get('params', {}))
  - Store env[node.output] = result
- Return env.get('final') or the last produced output

Advantages:

- Supports DAGs
- Supports fanout
- Eliminates left/right state machine for joins
- Better error messages

---

## Migration plan

1. Define node schema and operator signature in code and tests.
2. Add an adapter layer that wraps existing ops to the new signature (thin wrappers).
3. Implement new execute method in TQPExecutor that consumes the new plan nodes.
4. Update planner/compiler to emit the new node schema (or write a translator from old plan to new plan).
5. Run and extend tests.

---

## Testing and validation recommendations

- Unit tests for:
  - Binary joins where left and right come from non-consecutive scans.
  - Fanout: one scan reused by two downstream ops.
  - Group_by with multiple agg_exprs.
  - Error cases: missing input, missing params, wrong arity.
- Add property tests for commutativity where applicable (e.g., sort idempotence).
- Integration tests that run full compiler -> executor pipeline.

---

## Example tests to add

- test_join_with_nonadjacent_inputs
- test_reuse_scan_for_multiple_ops
- test_group_by_multiple_aggs
- test_plan_validation_errors

### Complete workflow for Filter example
 mÒ¿efp'[pkafafasfafad[fp[fksdfdf]]]
```
┌─────────────────────────────────────────────────────────────┐
│ INPUT: Spark Physical Plan                                  │
│   Filter(l_quantity < 24)                                   │
│     └── FileScan(lineitem)                                  │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ LAYER 1: SparkPhysicalPlanParser.parse()                    │
│   - DFS post-order traversal                                │
│   - Convert to TQP IR nodes                                 │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ TQP IR Graph:                                               │
│   IRNode(FILTER, condition=Expression(...))                 │
│     └── IRNode(SCAN, table='lineitem')                      │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ LAYER 2: IROptimizer.canonicalize() + optimize()            │
│   - Remove redundant operators                              │
│   - Apply optimization rules                                │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ Optimized IR (same structure in this simple case)           │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ LAYER 3: TQPPlanner.plan()                                  │
│   - DFS post-order traversal                                │
│   - Fetch tensor programs from operator_dict                │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ Operator Plan (List of Tensor Programs):                    │
│   1. ('scan', RelationalOperators.scan, {table: 'lineitem'})│
│   2. ('filter', RelationalOperators.filter, {condition: ...}│
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ LAYER 4: TQPExecutor(plan)                                  │
│   - Wraps plan in executor object                           │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ EXECUTION: executor.execute(tables)                         │
│                                                             │
│ Step 1: Execute SCAN                                        │
│   result = tables['lineitem']                               │
│   → TensorTable(5 rows, 3 columns as tensors)               │
│                                                             │
│ Step 2: Execute FILTER                                      │
│   a) Compile expression (post-order DFS):                   │
│      - Get column: tensor([10, 25, 30, 15, 45])             │
│      - Get literal: tensor([24, 24, 24, 24, 24])            │
│      - Apply torch.lt: tensor([T, F, F, T, F])              │
│   b) Apply mask with torch.masked_select                    │
│   c) Garbage collect old result                             │
│   → TensorTable(2 rows, filtered)                           │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ OUTPUT: TensorTable                                         │
│   l_quantity: tensor([10., 15.])                            │
│   l_orderkey: tensor([1., 4.])                              │
└─────────────────────────────────────────────────────────────┘
```

## Key Execution Patterns

1. DFS Post-Order Traversal used in:

- Layer 1: Parsing Spark plan to IR
- Layer 3: Building operator plan from IR
- Expression compilation

2. Topological Order Execution:

- Executor processes operators in order from plan
- Outputs are wired as inputs to next operator

3. Automatic Resource Management:

- Unused tensors are garbage collected via Python reference counting
- No manual memory management needed

4. Dictionary-Based Dispatch:

- operator_dict maps IR operators to tensor programs
- Enables extensibility (add new operators easily)

---

## Extension points

- Add a `plan optimizer` to inline trivial projections or push down filters.
- Add target backends (TorchScript, ONNX) once operators have deterministic signatures and are serializable.
- Add resource hints per node (memory estimate, prefer_copy vs prefer_view).

---

## Closing notes

The current TQPExecutor is a lightweight, sequential executor that suffices for strictly linear plans. It should be migrated to a named intermediate model and adopt a fixed operator contract to support real query shapes, robust error handling, and easier maintainability.
