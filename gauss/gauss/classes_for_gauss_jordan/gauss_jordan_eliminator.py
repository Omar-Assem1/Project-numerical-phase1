# classes_for_gauss_jordan/gj.py
import math


class GaussJordanEliminator:
    def __init__(self, augmented_matrix, precision=6):
        self.n = len(augmented_matrix)
        self.M = [[self.round_sig(val, precision) for val in row] for row in augmented_matrix]
        self.rank = 0
        self.pivot_positions = []
        self.precision = precision

    def round_sig(self, x, sig=None):
        if sig is None:
            sig = self.precision
        if x == 0:
            return 0.0
        order = math.floor(math.log10(abs(x)))
        factor = 10 ** (order - sig + 1)
        return round(x / factor) * factor

    def eliminate(self, steps):
        steps.append("\n" + "=" * 80)
        steps.append("       GAUSS-JORDAN ELIMINATION → REDUCED ROW ECHELON FORM")
        steps.append("                  with significant-figure rounding")
        steps.append("=" * 80)
        self._print("Initial Augmented Matrix", steps)

        h = 0
        for col in range(self.n):
            # Find pivot
            pivot_row = None
            for i in range(h, self.n):
                if abs(self.M[i][col]) > 1e-10:
                    if pivot_row is None or abs(self.M[i][col]) > abs(self.M[pivot_row][col]):
                        pivot_row = i

            if pivot_row is None:
                continue

            # Swap
            if pivot_row != h:
                steps.append(f"\nSwap Row{h + 1} ↔ Row{pivot_row + 1}")
                self.M[h], self.M[pivot_row] = self.M[pivot_row], self.M[h]
                self._print("After swap", steps)

            # Normalize pivot row
            pivot = self.M[h][col]
            pivot_r = self.round_sig(pivot, self.precision)
            steps.append(f"\nRow{h + 1} ÷ {pivot_r}  →  make pivot = 1")
            for j in range(self.n + 1):
                self.M[h][j] = self.round_sig(self.M[h][j] / pivot)
            self._print("After normalizing pivot row", steps)

            # Eliminate above and below
            for i in range(self.n):
                if i == h or abs(self.M[i][col]) < 1e-10:
                    continue

                factor = self.round_sig(self.M[i][col])
                factor_r = self.round_sig(factor, self.precision)
                steps.append(f"Row{i + 1} -= ({factor_r}) × Row{h + 1}")

                for j in range(self.n + 1):
                    product = self.round_sig(factor * self.M[h][j])
                    self.M[i][j] = self.round_sig(self.M[i][j] - product)

                self.M[i][col] = 0.0
                self._print(f"After eliminating in Row{i + 1}", steps)

            self.pivot_positions.append((h, col))
            h += 1

        self.rank = h
        steps.append("\n" + "=" * 80)
        steps.append(f"           REDUCED ROW ECHELON FORM (RREF) – RANK = {self.rank}")
        steps.append("=" * 80)
        self._print("Final RREF", steps)

    def _print(self, msg, steps):
        if msg:
            steps.append(f"\n{msg}:")
        for i, row in enumerate(self.M):
            rounded = [self.round_sig(x, self.precision) for x in row]
            row_str = "  ".join(f"{x:12.6g}" for x in rounded)
            steps.append(f"R{i + 1}: {row_str}")
        steps.append("")

    def get_rref_result(self):
        return self.M, self.rank, self.pivot_positions