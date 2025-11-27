import math


class ForwardEliminator:
    """Performs Gaussian elimination with partial pivoting and records all steps"""
    def __init__(self, augmented_matrix, precision=4):
        self.n = len(augmented_matrix)
        self.M = [row[:] for row in augmented_matrix]
        self.rank = 0
        self.pivot_positions = []
        self.precision = precision

    def round_sig(self, x, sig):
        if x == 0:
            return 0.0
        order = math.floor(math.log10(abs(x)))
        factor = 10 ** (order - sig + 1)
        return round(x / factor) * factor

    def eliminate(self, steps):
        steps.append("\n" + "=" * 75)
        steps.append("        FORWARD ELIMINATION WITH PARTIAL PIVOTING")
        steps.append("=" * 75)
        steps.append(f"Precision: {self.precision} significant figures")
        self._print("Initial Matrix", steps)

        row = 0
        step_count = 0

        for col in range(self.n):
            # Find pivot
            pivot_idx = None
            for i in range(row, self.n):
                if abs(self.M[i][col]) > 1e-10:
                    if pivot_idx is None or abs(self.M[i][col]) > abs(self.M[pivot_idx][col]):
                        pivot_idx = i

            if pivot_idx is None:
                continue

            # Swap rows
            if pivot_idx != row:
                steps.append(f"\nSWAP: Row{row + 1} ↔ Row{pivot_idx + 1}")
                steps.append("=" * 75)
                self.M[row], self.M[pivot_idx] = self.M[pivot_idx], self.M[row]
                self._print("After swap", steps)

            pivot = self.M[row][col]
            pivot_rounded = self.round_sig(pivot, self.precision)

            steps.append(f"\nPIVOT at position ({row + 1},{col + 1}) = {pivot_rounded}")
            steps.append("=" * 75)

            # Eliminate below
            for i in range(row + 1, self.n):
                if abs(self.M[i][col]) > 1e-10:
                    step_count += 1

                    numerator = self.M[i][col]
                    denominator = self.M[row][col]
                    factor = numerator / denominator
                    factor_rounded = self.round_sig(factor, self.precision)

                    steps.append(f"\n--- Step {step_count}: Row{i + 1} -= factor × Row{row + 1} ---")
                    steps.append(f"Factor = M[{i + 1}][{col + 1}] / M[{row + 1}][{col + 1}]")
                    steps.append(f"       = {self.round_sig(numerator, self.precision)} / {self.round_sig(denominator, self.precision)}")
                    steps.append(f"       = {factor_rounded}")
                    steps.append(f"Operation: Row{i + 1} = Row{i + 1} - ({factor_rounded}) × Row{row + 1}")
                    steps.append("-" * 75)

                    for j in range(col, self.n + 1):
                        old_val = self.M[i][j]
                        pivot_row_val = self.M[row][j]
                        product = factor_rounded * pivot_row_val
                        new_val = old_val - product

                        old_val_r = self.round_sig(old_val, self.precision)
                        pivot_row_val_r = self.round_sig(pivot_row_val, self.precision)
                        product_r = self.round_sig(product, self.precision)
                        new_val_r = self.round_sig(new_val, self.precision)

                        if j == col:
                            steps.append(f"M[{i + 1}][{j + 1}]: {old_val_r} - ({factor_rounded} × {pivot_row_val_r}) = 0")
                            self.M[i][j] = 0.0
                        else:
                            steps.append(f"M[{i + 1}][{j + 1}]: {old_val_r} - ({factor_rounded} × {pivot_row_val_r}) = {new_val_r}")
                            self.M[i][j] = new_val_r

                    self._print(f"Matrix after Row{i + 1} operation", steps)

            self.pivot_positions.append((row, col))
            row += 1

        self.rank = row

        steps.append("\n" + "=" * 75)
        steps.append(f"           ROW ECHELON FORM  →  RANK = {self.rank}")
        steps.append("=" * 75)
        self._print("Final Echelon Matrix", steps)

    def _print(self, msg, steps):
        if msg:
            steps.append(f"\n{msg}:")
        for i, r in enumerate(self.M):
            row_str = "  ".join(f"{self.round_sig(x, self.precision):12.6g}" for x in r)
            steps.append(f"R{i + 1}: {row_str}")
        steps.append("")

    def get_result(self):
        return self.M, self.rank, self.pivot_positions