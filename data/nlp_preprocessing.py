import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        words=set()
        for sentence in positive+negative:
            words.update(sentence.split())
        vocab={word:i+1 for i,word in enumerate(sorted(words))}

        def encode(sentence):
            return torch.tensor([vocab[i] for i in sentence.split()], dtype=torch.float)

        tensors=[encode(s) for s in positive+negative]
        return nn.utils.rnn.pad_sequence(tensors, batch_first=True)