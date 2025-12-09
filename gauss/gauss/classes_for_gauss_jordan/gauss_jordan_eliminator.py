# classes_for_gauss_jordan/gauss_jordan_eliminator.py
import math
from decimal import Context


class GaussJordanEliminator:
    def __init__(self, augmented_matrix, precision=6):
        self.n = len(augmented_matrix)
        self.M = [row[:] for row in augmented_matrix]  # Keep original precision
        self.rank = 0
        self.pivot_positions = []
        self.precision = precision

        self.step_strings = []
        self._current = []

    def round_sig(self, x):
        # Create a context with the desired precision
        ctx = Context(prec=self.precision)
        # Normalize applies the precision to the number
        return float(ctx.create_decimal(x).normalize())

    def _flush(self):
        if self._current:
            self.step_strings.append("\n".join(self._current))
            self._current = []

    def _print(self, msg=""):
        if msg:
            self._current.append(f"{msg}:")
        for i, row in enumerate(self.M):
            rounded = [self.round_sig(x) for x in row]
            row_str = "  ".join(f"{x:12.6g}" for x in rounded)
            self._current.append(f"R{i + 1}: {row_str}")
        self._current.append("")

    def eliminate(self):
        # Step 0: Header + Initial Matrix
        header = [
            "=" * 80,
            "       GAUSS-JORDAN ELIMINATION → REDUCED ROW ECHELON FORM",
            "                  with significant-figure rounding",
            "=" * 80,
            f"Precision: {self.precision} significant figures",
            "",
            "Initial Augmented Matrix:"
        ]
        self._current = header
        self._print()
        self._flush()

        h = 0
        for col in range(self.n):
            # Find pivot
            pivot_row = None
            max_val = 0
            for i in range(h, self.n):
                val = abs(self.M[i][col])
                if val > 1e-10 and (pivot_row is None or val > max_val):
                    max_val = val
                    pivot_row = i

            if pivot_row is None:
                continue

            # Swap
            if pivot_row != h:
                self._current = [
                    f"SWAP Row{h + 1} ↔ Row{pivot_row + 1}",
                    "=" * 80
                ]
                self.M[h], self.M[pivot_row] = self.M[pivot_row], self.M[h]
                self._print("After swap")
                self._flush()

            # Normalize pivot row (make pivot = 1)
            pivot = self.M[h][col]
            pivot_r = self.round_sig(pivot)
            self._current = [
                f"Row{h + 1} ÷ {pivot_r} → make pivot = 1",
                "-" * 80
            ]
            for j in range(self.n + 1):
                self.M[h][j] = self.round_sig(self.M[h][j] / pivot)
            self._print("After normalizing pivot row")
            self._flush()

            # Eliminate in all other rows (above and below)
            for i in range(self.n):
                if i == h or abs(self.M[i][col]) < 1e-10:
                    continue

                factor = self.M[i][col]
                factor_r = self.round_sig(factor)

                self._current = [
                    f"ELIMINATE COLUMN {col + 1} IN ROW {i + 1}",
                    f"Row{i + 1} -= ({factor_r}) × Row{h + 1}",
                    "-" * 80
                ]

                for j in range(self.n + 1):
                    self.M[i][j] = self.round_sig(self.M[i][j] - factor * self.M[h][j])
                self.M[i][col] = 0.0

                self._print(f"After eliminating in Row{i + 1}")
                self._flush()

            self.pivot_positions.append((h, col))
            h += 1

        self.rank = h

        # Final RREF
        final = [
            "=" * 80,
            f"           REDUCED ROW ECHELON FORM (RREF) – RANK = {self.rank}",
            "=" * 80,
            "Final RREF:"
        ]
        self._current = final
        self._print()
        self._flush()

    def get_rref_result(self):
        return self.M, self.rank, self.pivot_positions