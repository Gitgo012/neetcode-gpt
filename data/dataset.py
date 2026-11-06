import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]], List[List[str]]]:
        # 1. Tokenize by splitting on whitespace: raw_dataset.split()
        # 2. Generate batch_size random start indices using torch.randint()
        #    Range: [0, len(tokens) - context_length)
        # 3. For each index i, X = tokens[i:i+context_length], Y = tokens[i+1:i+1+context_length]
        torch.manual_seed(0)
        tokens = raw_dataset.split()
        # Generate random starting positions
        start_indices = torch.randint(
            0,
            len(tokens) - context_length,
            (batch_size,)
        )

        # Create input sequences (X)
        X = [
            tokens[i : i + context_length]
            for i in start_indices
        ]

        # Create target sequences (Y), shifted by one
        Y = [
            tokens[i + 1 : i + context_length + 1]
            for i in start_indices
        ]

        return X, Y
