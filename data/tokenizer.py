
from typing import List
from collections import Counter


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:

        # Initially, each character is a token
        tokens = list(corpus)

        merges = []

        for _ in range(num_merges):

            # Step 1: Count adjacent pairs
            pair_counts = Counter(
                (tokens[i], tokens[i + 1])
                for i in range(len(tokens) - 1)
            )

            # Stop if there are no adjacent pairs
            if not pair_counts:
                break

            # Step 2: Find most frequent pair
            # Lexicographically smallest pair wins ties
            best_pair = min(
                pair_counts,
                key=lambda pair: (-pair_counts[pair], pair)
            )

            token_a, token_b = best_pair

            # Step 3: Merge non-overlapping occurrences
            new_tokens = []
            i = 0

            while i < len(tokens):

                if (
                    i < len(tokens) - 1
                    and tokens[i] == token_a
                    and tokens[i + 1] == token_b
                ):
                    new_tokens.append(token_a + token_b)
                    i += 2

                else:
                    new_tokens.append(tokens[i])
                    i += 1

            # Step 4: Record the merge
            merges.append([token_a, token_b])

            tokens = new_tokens

        return merges


solution = Solution()

print(solution.get_merges("aaabdaaabac", 5))
