from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        result = []

        for number in numbers:

            text = str(number)
            tokens = []
            i = 0

            while i < len(text):

                # Find the longest matching token
                match = None

                for j in range(len(text), i, -1):

                    substring = text[i:j]

                    if substring in vocab:
                        match = substring
                        break

                if match is None:
                    raise ValueError(f"No matching token at position {i}")

                tokens.append(match)
                i += len(match)

            result.append(tokens)

        return result

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        count = 0
        i = 0

        while i < len(text):

            match = None

            for j in range(len(text), i, -1):

                substring = text[i:j]

                if substring in vocab:
                    match = substring
                    break

            if match is None:
                raise ValueError(f"No matching token at position {i}")

            count += 1
            i += len(match)

        return count

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        words=text.split()
        if not words:
            return 0.0
        total_tokens=self.count_tokens(text,vocab)
        return round(total_tokens/len(words),4)
