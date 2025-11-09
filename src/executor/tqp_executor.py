from typing import Callable, Dict, List, Tuple, Any, Union
from tensor import TensorTable

# Legacy plan element: Tuple[str, Callable, Dict]
LegacyPlanElem = Tuple[str, Callable, Dict]

# New plan node: Dict with keys: op, op_func, inputs, output, params
PlanNode = Dict[str, Any]
PlanElem = Union[LegacyPlanElem, PlanNode]


class TQPExecutor:
    """Executor for tensor-query plans.

    Supports two plan styles for backward compatibility:
    - Legacy linear tuples: (op_name, op_func, params) (keeps original sequential semantics)
    - New explicit nodes: {
          "op": "filter",
          "op_func": callable,
          "inputs": ["scan_users"],
          "output": "filtered_users",
          "params": {...}
      }

    New style is preferred: it uses a named-environment model (env) and
    a stable operator signature: Callable[[List[TensorTable], Dict], TensorTable].
    Legacy style is adapted to the new env model where possible.
    """

    def __init__(self, plan: List[PlanElem]):
        self.plan = plan
    
    def _resolve_legacy_input(self, last_name: str, params: Dict, env: Dict[str, TensorTable]) -> TensorTable:
        """Resolve input for legacy-style ops.

        Priority:
          1. last_name (previous result) if set
          2. params['table'] if provided (common for scan/filter emitted without a scan step)
          3. params['input'] if provided
        Raises a clear KeyError / RuntimeError if not found.
        """
        if last_name:
            return env[last_name]
        
        # Try common param names that legacy plans may use
        table_name = params.get("table") or params.get("input")
        if table_name:
            if table_name not in env:
                raise KeyError(f"Input table '{table_name}' not provided in input tables")
            return env[table_name]
        raise RuntimeError("No input available for legacy op (no previous result and no 'table'/'input' param)")

    def _call_op_flexible(self, op_func: Callable, inputs: List[TensorTable], params: Dict) -> TensorTable:
        """Call operator with flexible adapters to support both new and legacy op signatures.

        Preferred operator signature:
          op_func(inputs: List[TensorTable], params: Dict) -> TensorTable

        Fallbacks tried in order:
          - op_func(inputs, params)
          - op_func(*inputs, **params)
          - op_func(*inputs)
        """
        # Preferred: explicit list + params
        try:
            return op_func(inputs, params)
        except TypeError:
            pass

        # Try positional expansion with params as kwargs
        try:
            return op_func(*inputs, **params)
        except TypeError:
            pass

        # Try positional-only (legacy)
        try:
            return op_func(*inputs)
        except TypeError as e:
            # Provide a clearer error
            raise TypeError(f"Operator call failed for {op_func}: {e}")

    def execute(self, tables: Dict[str, TensorTable]) -> TensorTable:
        """Execute plan using a named environment.

        - `tables` provides initial named inputs available to the plan.
        - Plan nodes populate `env` with named intermediates.
        - Returns env['final'] if present, else the last produced result.

        Raises clear exceptions on validation problems.
        """
        if not self.plan:
            raise ValueError("Empty execution plan")

        # Environment of named tables/results: seed with input tables (copies of references)
        env: Dict[str, TensorTable] = dict(tables)

        tmp_counter = 0
        last_name: str = ""  # name of last produced result

        for elem in self.plan:
            # New-style node: dict with explicit inputs/outputs
            if isinstance(elem, dict):
                op_name = elem.get("op")
                op_func = elem.get("op_func")
                inputs_names = elem.get("inputs", [])
                output_name = elem.get("output")
                params = elem.get("params", {})

                if op_func is None:
                    raise ValueError(f"Plan node missing 'op_func': {elem}")

                # Validate inputs exist
                missing = [n for n in inputs_names if n not in env]
                if missing:
                    raise KeyError(f"Missing input(s) for node '{op_name}': {missing}")

                inputs_vals = [env[n] for n in inputs_names]
                result = self._call_op_flexible(op_func, inputs_vals, params)

                if not output_name:
                    # assign generated name if output not provided
                    output_name = f"__tmp{tmp_counter}"
                    tmp_counter += 1

                env[output_name] = result
                last_name = output_name

            # Legacy-style tuple: (op_name, op_func, params)
            elif isinstance(elem, tuple) and len(elem) == 3:
                print("elem = ", elem)
                op_name, op_func, params = elem
                if not isinstance(params, dict):
                    raise ValueError(f"Legacy plan params must be a dict for op {op_name}")

                # Handle legacy ops with their original calling convention,
                # but always register results in env under a generated name.
                output_name = f"__tmp{tmp_counter}"
                tmp_counter += 1

                if op_name == "scan":
                    # resolve input from previous result or params['table']/['input']
                    input_val = self._resolve_legacy_input(last_name, params, env)
                    result = self._call_op_flexible(op_func, [input_val], params)
                    # store under generated name as well (so downstream nodes can reference it if migrated)
                    env[output_name] = result
                    last_name = output_name

                elif op_name == "filter":
                    # resolve input from previous result or params['table']/['input']
                    input_val = self._resolve_legacy_input(last_name, params, env)
                    # legacy filter signature: op_func(table, condition)
                    condition = params.get("condition")
                    if condition is None:
                        raise KeyError("filter params must include 'condition'")
                    try:
                        result = op_func(input_val, condition)
                    except TypeError:
                        # fallback to flexible adapter
                        result = self._call_op_flexible(op_func, [input_val], params)
                    env[output_name] = result
                    last_name = output_name

                elif op_name == "project":
                    input_val = self._resolve_legacy_input(last_name, params, env)
                    columns = params.get("columns")
                    if columns is None:
                        raise KeyError("project params must include 'columns'")
                    try:
                        result = op_func(input_val, columns)
                    except TypeError:
                        result = self._call_op_flexible(op_func, [input_val], params)
                    env[output_name] = result
                    last_name = output_name

                elif op_name == "sort":
                    input_val = self._resolve_legacy_input(last_name, params, env)
                    key = params.get("key")
                    ascending = params.get("ascending", True)
                    if key is None:
                        raise KeyError("sort params must include 'key'")
                    try:
                        result = op_func(input_val, key, ascending)
                    except TypeError:
                        result = self._call_op_flexible(op_func, [input_val], params)
                    env[output_name] = result
                    last_name = output_name

                elif op_name in ("sort_join", "hash_join"):
                    # Legacy join relied on two sequential inputs: current_left then current_right.
                    # We emulate that by remembering a special pending join slot in env.
                    pending_key = "__pending_join"
                    if pending_key not in env:
                        # store left side
                        if not last_name:
                            raise RuntimeError(f"{op_name} left input missing")
                        env[pending_key] = env[last_name]
                        # do not produce a new tmp result yet; continue to next plan element
                        # mark last_name as empty so next op is considered next input
                        last_name = ""
                    else:
                        # have left, use current last_name as right
                        if not last_name:
                            raise RuntimeError(f"{op_name} right input missing")
                        left = env.pop(pending_key)
                        right = env[last_name]
                        left_key = params.get("left_key")
                        right_key = params.get("right_key")
                        if left_key is None or right_key is None:
                            raise KeyError(f"{op_name} params must include 'left_key' and 'right_key'")
                        try:
                            result = op_func(left, right, left_key, right_key)
                        except TypeError:
                            result = self._call_op_flexible(op_func, [left, right], params)
                        env[output_name] = result
                        last_name = output_name

                elif op_name == "group_by":
                    input_val = self._resolve_legacy_input(last_name, params, env)
                    agg_exprs = params.get("agg_exprs")
                    group_cols = params.get("group_cols")
                    if not agg_exprs or not isinstance(agg_exprs, list):
                        raise KeyError("group_by params must include 'agg_exprs' list")
                    # Legacy used first agg only
                    agg_expr = agg_exprs[0]
                    agg_col = agg_expr.get("column")
                    agg_fn = agg_expr.get("function")
                    if group_cols is None or agg_col is None or agg_fn is None:
                        raise KeyError("group_by params missing required keys")
                    try:
                        result = op_func(input_val, group_cols, agg_col, agg_fn)
                    except TypeError:
                        result = self._call_op_flexible(op_func, [input_val], params)
                    env[output_name] = result
                    last_name = output_name

                else:
                    # Generic fallback for unknown legacy op: attempt flexible call with last result as input if present
                    inputs_vals = []
                    if last_name:
                        inputs_vals = [env[last_name]]
                    try:
                        result = self._call_op_flexible(op_func, inputs_vals, params)
                    except TypeError as e:
                        raise TypeError(f"Failed to call legacy op '{op_name}': {e}")
                    env[output_name] = result
                    last_name = output_name


            else:
                raise TypeError(f"Unsupported plan element type: {type(elem)}")

        # Prefer explicitly named final result
        if "final" in env:
            return env["final"]

        if last_name and last_name in env:
            return env[last_name]

        # Nothing produced
        raise RuntimeError("Plan executed but produced no results")

"""
plan =  [('filter', <function RelationalOperators.filter at 0x7d8a03635d00>, {'condition': Expression(expr_type=<ExprType.LT: 'lt'>, value=None, children=[Expression(expr_type=<ExprType.COLUMN: 'column'>, value='l_quantity', children=[]), Expression(expr_type=<ExprType.LITERAL: 'literal'>, value=24.0, children=[])])})]
elem =  ('filter', <function RelationalOperators.filter at 0x7d8a03635d00>, {'condition': Expression(expr_type=<ExprType.LT: 'lt'>, value=None, children=[Expression(expr_type=<ExprType.COLUMN: 'column'>, value='l_quantity', children=[]), Expression(expr_type=<ExprType.LITERAL: 'literal'>, value=24.0, children=[])])})
---------------------------------------------------------------------------
"""