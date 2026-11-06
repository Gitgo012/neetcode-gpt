import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.

        # Initialize AdamW optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        final_loss = 0.0

        for epoch in range(epochs):

            # Set seed for reproducibility
            torch.manual_seed(epoch)

            # Generate random starting indices
            start_indices = torch.randint(
                0,
                len(data) - context_length,
                (batch_size,)
            )

            # Create input and target batches
            X = torch.stack([
                data[i : i + context_length]
                for i in start_indices
            ])

            Y = torch.stack([
                data[i + 1 : i + context_length + 1]
                for i in start_indices
            ])

            # Forward pass
            logits = model(X)

            # Calculate cross entropy loss
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                Y.reshape(-1)
            )

            # Clear previous gradients
            optimizer.zero_grad()

            # Backpropagation
            loss.backward()

            # Update model parameters
            optimizer.step()

            final_loss = loss.item()

        return round(final_loss, 4)
