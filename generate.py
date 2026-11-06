import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution:
    def generate(self, model: nn.Module, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict) -> str:
        generator = torch.manual_seed(0)
        initial_state = generator.get_state()
        result = ""
        
        for i in range(new_chars):
            # 1. Crop context to context_length
            context = context[:, -context_length:]

            # 2. Run model to get logits and take the last position
            logits = model(context)
            logits = logits[:, -1, :] # Shape: (batch_size, vocab_size)

            # 3. Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)

            # Optional: if strict test harness requires resetting generator state per step
            # generator.set_state(initial_state)

            # 4. Sample next token using multinomial
            next_token = torch.multinomial(
                probs,
                num_samples=1,
                generator=generator
            )

            # 5. Append sampled token to context
            context = torch.cat((context, next_token), dim=1)

            # 6. Map token ID to character and accumulate
            result += int_to_char[next_token.item()]

        return result