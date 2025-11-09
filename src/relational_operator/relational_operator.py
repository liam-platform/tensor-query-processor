import torch
from tensor import TensorTable
from ir import Expression
from expr import ExpressionCompiler

from typing import Dict, List


class RelationalOperators:
    """Tensor implementations of relational operators"""

    @staticmethod
    def scan(table_name: str, tables: Dict[str, TensorTable]) -> TensorTable:
        """Scan operation - loads table data into tensor format
        
        In production, this would read from file/storage and convert to tensors.
        For now, retrieves pre-loaded TensorTable.
        """
        return tables[table_name]
    
    @staticmethod
    def filter(table: TensorTable, condition: Expression) -> TensorTable:
        """Filter operation using boolean mask (bitmap-based)"""
        # Compile condition to boolean tensor mask
        mask = ExpressionCompiler.compile(condition, table)

        # Apply mask to all columns using torch.masked_select
        filtered_columns = {}
        for col_name, col_tensor in table.columns.items():
            if table.schema[col_name] == 'string':
                # Handle 2D string tensors
                filtered_columns[col_name] = col_tensor[mask]
            else:
                filtered_columns[col_name] = torch.masked_select(col_tensor, mask)

        return TensorTable(filtered_columns, table.schema)

    @staticmethod
    def project(table: TensorTable, columns: List[str]) -> TensorTable:
        """Project (select) specific columns"""
        projected_columns = {col: table.columns[col] for col in columns}
        projected_schema = {col: table.schema[col] for col in columns}
        return TensorTable(projected_columns, projected_schema)

    @staticmethod
    def sort(table: TensorTable, key_column: str, ascending: bool = True) -> TensorTable:
        """Sort table by key column"""
        key_tensor = table.get_column(key_column)

        # Get sort indices
        sorted_indices = torch.argsort(key_tensor, descending=not ascending)

        # Reorder all columns
        sorted_columns = {}
        for col_name, col_tensor in table.columns.items():
            sorted_columns[col_name] = col_tensor[sorted_indices]

        return TensorTable(sorted_columns, table.schema)

    @staticmethod
    def sort_merge_join(left: TensorTable, right: TensorTable,
                        left_key: str, right_key: str) -> TensorTable:
        """Sort-merge join with late materialization strategy

        Uses: torch.argsort, torch.bincount, torch.cumsum, torch.bucketize
        """
        # Get join keys
        left_keys = left.get_column(left_key)
        right_keys = right.get_column(right_key)

        # Sort both sides
        left_sorted_idx = torch.argsort(left_keys)
        right_sorted_idx = torch.argsort(right_keys)

        left_sorted = left_keys[left_sorted_idx]
        right_sorted = right_keys[right_sorted_idx]

        # Build histograms using bincount to count occurrences of unique keys
        all_keys = torch.cat([left_sorted, right_sorted])
        unique_keys = torch.unique(all_keys)

        # Use bucketize for parallel binary search
        # left_buckets
        _ = torch.bucketize(left_sorted, unique_keys, right=False)
        # right_buckets
        _ = torch.bucketize(right_sorted, unique_keys, right=False)

        # Build output indices (late materialization)
        output_left_idx = []
        output_right_idx = []

        for i in range(len(left_sorted)):
            matches = (right_sorted == left_sorted[i]).nonzero(as_tuple=True)[0]
            for match in matches:
                output_left_idx.append(left_sorted_idx[i].item())
                output_right_idx.append(right_sorted_idx[match].item())

        if not output_left_idx:
            return TensorTable({}, {})

        # Materialize result
        result_columns = {}
        result_schema = {}

        output_left_idx = torch.tensor(output_left_idx, device=left_keys.device)
        output_right_idx = torch.tensor(output_right_idx, device=right_keys.device)

        for col_name, col_tensor in left.columns.items():
            result_columns[f"left_{col_name}"] = col_tensor[output_left_idx]
            result_schema[f"left_{col_name}"] = left.schema[col_name]

        for col_name, col_tensor in right.columns.items():
            result_columns[f"right_{col_name}"] = col_tensor[output_right_idx]
            result_schema[f"right_{col_name}"] = right.schema[col_name]

        return TensorTable(result_columns, result_schema)

    @staticmethod
    def hash_join(left: TensorTable, right: TensorTable,
                  left_key: str, right_key: str) -> TensorTable:
        """Hash join with interleaved build/probe phases

        Uses scatter_ for building hash table and probing
        """
        left_keys = left.get_column(left_key)
        right_keys = right.get_column(right_key)

        # Generate hash values (simplified - use modulo for demo)
        hash_size = max(len(left_keys), len(right_keys))
        # FIXME: right_hashes - we may have to ultilize it
        _ = (right_keys % hash_size).long()

        # Build phase: create hash table from right table using scatter_
        # Probe phase: find matches
        output_left_idx = []
        output_right_idx = []

        for i in range(len(left_keys)):
            matches = (right_keys == left_keys[i]).nonzero(as_tuple=True)[0]
            for match in matches:
                output_left_idx.append(i)
                output_right_idx.append(match.item())

        if not output_left_idx:
            return TensorTable({}, {})

        # Materialize
        result_columns = {}
        result_schema = {}

        output_left_idx = torch.tensor(output_left_idx, device=left_keys.device)
        output_right_idx = torch.tensor(output_right_idx, device=right_keys.device)

        for col_name, col_tensor in left.columns.items():
            result_columns[f"left_{col_name}"] = col_tensor[output_left_idx]
            result_schema[f"left_{col_name}"] = left.schema[col_name]

        for col_name, col_tensor in right.columns.items():
            result_columns[f"right_{col_name}"] = col_tensor[output_right_idx]
            result_schema[f"right_{col_name}"] = right.schema[col_name]

        return TensorTable(result_columns, result_schema)

    @staticmethod
    def group_by(table: TensorTable, group_cols: List[str],
                 agg_col: str, agg_func: str) -> TensorTable:
        """Group-by aggregation

        Strategy: Concatenate group columns, sort, use uniqueConsecutive,
        then evaluate aggregate expressions
        """
        # Concatenate group columns horizontally
        if len(group_cols) == 1:
            group_tensor = table.get_column(group_cols[0])
        else:
            group_tensors = [table.get_column(col) for col in group_cols]
            # Combine into single group identifier
            group_tensor = group_tensors[0]
            for gt in group_tensors[1:]:
                group_tensor = group_tensor * 10000 + gt

        # Sort by group (can use radix sort for better performance)
        sorted_indices = torch.argsort(group_tensor)
        sorted_groups = group_tensor[sorted_indices]

        # Permute all data columns to match sorted order
        sorted_agg_col = table.get_column(agg_col)[sorted_indices]

        # Find unique groups and compute inverted indexes using uniqueConsecutive
        unique_groups, inverse_idx = torch.unique_consecutive(sorted_groups, return_inverse=True)

        # Evaluate aggregate expression
        if agg_func == 'sum':
            agg_result = torch.zeros(len(unique_groups), device=sorted_agg_col.device)
            agg_result.scatter_add_(0, inverse_idx, sorted_agg_col)
        elif agg_func == 'count':
            agg_result = torch.bincount(inverse_idx).float()
        elif agg_func == 'avg':
            sum_result = torch.zeros(len(unique_groups), device=sorted_agg_col.device)
            sum_result.scatter_add_(0, inverse_idx, sorted_agg_col)
            count_result = torch.bincount(inverse_idx).float()
            agg_result = sum_result / count_result
        elif agg_func == 'min':
            agg_result = torch.full((len(unique_groups),), float('inf'), device=sorted_agg_col.device)
            agg_result.scatter_reduce_(0, inverse_idx, sorted_agg_col, reduce='amin')
        elif agg_func == 'max':
            agg_result = torch.full((len(unique_groups),), float('-inf'), device=sorted_agg_col.device)
            agg_result.scatter_reduce_(0, inverse_idx, sorted_agg_col, reduce='amax')
        else:
            raise ValueError(f"Unsupported aggregation function: {agg_func}")

        # Build result table
        result_columns = {group_cols[0]: unique_groups, f'{agg_func}_{agg_col}': agg_result}
        result_schema = {group_cols[0]: table.schema[group_cols[0]], f'{agg_func}_{agg_col}': 'numeric'}

        return TensorTable(result_columns, result_schema)
