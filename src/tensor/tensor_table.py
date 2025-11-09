from typing import Dict, List
import torch


class TensorTable:
    """Columnar tensor representation of a relational table"""
    def __init__(self, columns: Dict[str, torch.Tensor], schema: Dict[str, str]):
        self.columns = columns
        self.schema = schema
        self.num_rows = len(next(iter(columns.values()))) if columns else 0

    def __len__(self):
        return self.num_rows

    def get_column(self, name: str) -> torch.Tensor:
        return self.columns[name]
    
    def get_all_columns(self) -> List[torch.Tensor]:
        return list(self.columns.values())

    @staticmethod
    def from_dict(data: Dict[str, List], device='cpu') -> 'TensorTable':
        """Convert dictionary of lists to TensorTable"""
        columns = {}
        schema = {}
        
        for col_name, col_data in data.items():
            if isinstance(col_data[0], str):
                # String encoding: n×m tensor
                max_len = max(len(s) for s in col_data)
                tensor = torch.zeros((len(col_data), max_len), dtype=torch.long, device=device)
                for i, s in enumerate(col_data):
                    for j, c in enumerate(s):
                        tensor[i, j] = ord(c)
                schema[col_name] = 'string'
            elif isinstance(col_data[0], (int, float)):
                tensor = torch.tensor(col_data, dtype=torch.float32, device=device)
                schema[col_name] = 'numeric'
            else:
                raise ValueError(f"Unsupported data type for column {col_name}")
            
            columns[col_name] = tensor
        
        return TensorTable(columns, schema)
