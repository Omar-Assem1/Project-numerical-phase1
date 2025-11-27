import math
class ForwardEliminatorScaling:
    """Gaussian elimination with scaled partial pivoting — records all steps"""
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
        steps.append("\n" + "=" * 80)
        steps.append("    GAUSSIAN ELIMINATION WITH SCALED PARTIAL PIVOTING")
        steps.append("=" * 80)
        steps.append(f"Precision: {self.precision} significant figures")
        self._print("Initial Augmented Matrix", steps)

        # Scale factors
        scale = [0.0] * self.n
        steps.append("\n" + "-" * 80)
        steps.append("COMPUTING SCALE FACTORS (max absolute value in each row)")
        steps.append("-" * 80)

        for i in range(self.n):
            row_max = max(abs(self.M[i][j]) for j in range(self.n))
            scale[i] = row_max if row_max > 0 else 1.0
            scale[i] = self.round_sig(scale[i], self.precision)
            steps.append(f"Scale[Row{i + 1}] = max|M[{i + 1}][j]| = {scale[i]}")
        steps.append("")

        row = 0
        step_count = 0

        for col in range(self.n):
            if row >= self.n:
                break

            steps.append(f"\nFINDING PIVOT FOR COLUMN {col + 1}")
            steps.append("=" * 80)

            best_idx = row
            best_ratio = -1.0

            steps.append("\nCalculating scaled ratios:")
            for i in range(row, self.n):
                ratio = abs(self.M[i][col]) / scale[i] if scale[i] > 0 else 0.0
                ratio_r = self.round_sig(ratio, self.precision)
                marker = " ← BEST" if ratio_r > best_ratio else ""
                if marker:
                    best_ratio = ratio_r
                    best_idx = i
                steps.append(f"  Row{i + 1}: |{self.round_sig(self.M[i][col], self.precision)}| / {scale[i]} = {ratio_r}{marker}")

            if best_ratio < 1e-12:
                steps.append(f"\nNo valid pivot found in column {col + 1}, skipping...")
                continue

            steps.append(f"\nSelected pivot: Row{best_idx + 1} (scaled ratio = {best_ratio})")

            if best_idx != row:
                steps.append(f"\nSWAP: Row{row + 1} ↔ Row{best_idx + 1}")
                steps.append("=" * 80)
                self.M[row], self.M[best_idx] = self.M[best_idx], self.M[row]
                scale[row], scale[best_idx] = scale[best_idx], scale[row]
                self._print("After swap", steps)

            pivot = self.M[row][col]
            pivot_rounded = self.round_sig(pivot, self.precision)
            steps.append(f"\nPIVOT at Row{row + 1}, Col{col + 1} = {pivot_rounded}")
            steps.append(f"Scaled ratio = {best_ratio}")
            steps.append("=" * 80)

            for i in range(row + 1, self.n):
                if abs(self.M[i][col]) > 1e-12:
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
                    steps.append("-" * 80)

                    for j in range(col, self.n + 1):
                        old_val = self.M[i][j]
                        pivot_row_val = self.M[row][j]
                        product = factor_rounded * pivot_row_val
                        new_val = old_val - product

                        old_r = self.round_sig(old_val, self.precision)
                        piv_r = self.round_sig(pivot_row_val, self.precision)
                        prod_r = self.round_sig(product, self.precision)
                        new_r = self.round_sig(new_val, self.precision)

                        if j == col:
                            steps.append(f"M[{i + 1}][{j + 1}]: {old_r} - ({factor_rounded} × {piv_r}) = 0")
                            self.M[i][j] = 0.0
                        else:
                            steps.append(f"M[{i + 1}][{j + 1}]: {old_r} - ({factor_rounded} × {piv_r}) = {new_r}")
                            self.M[i][j] = new_r

                    # Update scale
                    row_max = max(abs(self.M[i][j]) for j in range(self.n))
                    if row_max > 0:
                        scale[i] = self.round_sig(row_max, self.precision)
                        steps.append(f"\nUpdated Scale[Row{i + 1}] = {scale[i]}")

                    self._print(f"Matrix after Row{i + 1} operation", steps)

            self.pivot_positions.append((row, col))
            row += 1

        self.rank = row
        steps.append("\n" + "=" * 80)
        steps.append(f"           UPPER TRIANGULAR (ROW ECHELON) FORM")
        steps.append(f"                    RANK = {self.rank}")
        steps.append("=" * 80)
        self._print("Final Echelon Matrix", steps)

    def _print(self, message, steps):
        if message:
            steps.append(f"\n{message}:")
        for i, r in enumerate(self.M):
            row_str = "  ".join(f"{self.round_sig(x, self.precision):12.6g}" for x in r)
            steps.append(f"R{i + 1}: {row_str}")
        steps.append("")

    def get_result(self):
        return self.M, self.rank, self.pivot_positions