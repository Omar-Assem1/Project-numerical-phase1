import math
from decimal import Context

class ForwardEliminatorScaling:
    """Gaussian elimination with scaled partial pivoting — one big string per step"""
    def __init__(self, augmented_matrix, precision=4):
        self.n = len(augmented_matrix)
        self.M = [row[:] for row in augmented_matrix]
        self.rank = 0
        self.pivot_positions = []
        self.precision = precision

        # One string per logical step
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

    def _print(self, message=""):
        if message:
            self._current.append(f"{message}:")
        for i, r in enumerate(self.M):
            row_str = "  ".join(f"{self.round_sig(x):12.6g}" for x in r)
            self._current.append(f"R{i + 1}: {row_str}")
        self._current.append("")

    def eliminate(self):
        # === STEP 0: Header + Initial matrix + Scales ===
        header = [
            "=" * 80,
            "    GAUSSIAN ELIMINATION WITH SCALED PARTIAL PIVOTING",
            "=" * 80,
            f"Precision: {self.precision} significant figures",
            "",
            "Initial Augmented Matrix:"
        ]
        for i, r in enumerate(self.M):
            row_str = "  ".join(f"{self.round_sig(x):12.6g}" for x in r)
            header.append(f"R{i + 1}: {row_str}")
        header.append("")

        # Compute and show scales
        scale = [0.0] * self.n
        header.append("-" * 80)
        header.append("COMPUTING SCALE FACTORS (max absolute value in each row)")
        header.append("-" * 80)
        for i in range(self.n):
            row_max = max(abs(self.M[i][j]) for j in range(self.n))
            scale[i] = row_max if row_max > 0 else 1.0
            scale[i] = self.round_sig(scale[i])
            header.append(f"Scale[Row{i + 1}] = max|M[{i + 1}][j]| = {scale[i]}")
        header.append("")

        self._current = header
        self._flush()

        row = 0
        step_count = 0

        for col in range(self.n):
            if row >= self.n:
                break

            self._current = []
            self._current.append(f"FINDING PIVOT FOR COLUMN {col + 1}")
            self._current.append("=" * 80)

            best_idx = row
            best_ratio = -1.0

            self._current.append("\nCalculating scaled ratios:")
            for i in range(row, self.n):
                ratio = abs(self.M[i][col]) / scale[i] if scale[i] > 0 else 0.0
                ratio_r = self.round_sig(ratio)
                marker = " ← BEST" if ratio_r > best_ratio else ""
                if marker:
                    best_ratio = ratio_r
                    best_idx = i
                self._current.append(f"  Row{i + 1}: |{self.round_sig(self.M[i][col])}| / {scale[i]} = {ratio_r}{marker}")

            if best_ratio < 1e-12:
                self._current.append(f"\nNo valid pivot found in column {col + 1}, skipping...")
                self._flush()
                continue

            self._current.append(f"\nSelected pivot: Row{best_idx + 1} (scaled ratio = {best_ratio})")

            # Swap if needed
            if best_idx != row:
                self._current.append(f"\nSWAP: Row{row + 1} ↔ Row{best_idx + 1}")
                self._current.append("=" * 80)
                self.M[row], self.M[best_idx] = self.M[best_idx], self.M[row]
                scale[row], scale[best_idx] = scale[best_idx], scale[row]
                self._print("After swap")
                self._flush()
                self._current = []  # next steps start fresh

            pivot = self.M[row][col]
            pivot_rounded = self.round_sig(pivot)
            self._current.append(f"\nPIVOT at Row{row + 1}, Col{col + 1} = {pivot_rounded}")
            self._current.append(f"Scaled ratio = {best_ratio}")
            self._current.append("=" * 80)

            # Elimination below
            for i in range(row + 1, self.n):
                if abs(self.M[i][col]) <= 1e-12:
                    continue

                step_count += 1
                self._current = []  # ← new elimination step starts

                numerator = self.M[i][col]
                denominator = self.M[row][col]
                factor = numerator / denominator
                factor_rounded = self.round_sig(factor)

                self._current.extend([
                    f"--- Step {step_count}: Row{i + 1} -= factor × Row{row + 1} ---",
                    f"Factor = M[{i + 1}][{col + 1}] / M[{row + 1}][{col + 1}]",
                    f"       = {self.round_sig(numerator)} / {self.round_sig(denominator)}",
                    f"       = {factor_rounded}",
                    f"Operation: Row{i + 1} = Row{i + 1} - ({factor_rounded}) × Row{row + 1}",
                    "-" * 80
                ])

                for j in range(col, self.n + 1):
                    old_val = self.M[i][j]
                    pivot_row_val = self.M[row][j]
                    product = factor_rounded * pivot_row_val
                    new_val = old_val - product

                    old_r = self.round_sig(old_val)
                    piv_r = self.round_sig(pivot_row_val)
                    new_r = self.round_sig(new_val)

                    if j == col:
                        self._current.append(f"M[{i + 1}][{j + 1}]: {old_r} - ({factor_rounded} × {piv_r}) = 0")
                        self.M[i][j] = 0.0
                    else:
                        self._current.append(f"M[{i + 1}][{j + 1}]: {old_r} - ({factor_rounded} × {piv_r}) = {new_r}")
                        self.M[i][j] = new_r

                # Update scale for row i
                row_max = max(abs(self.M[i][j]) for j in range(self.n))
                if row_max > 0:
                    scale[i] = self.round_sig(row_max)
                    self._current.append(f"\nUpdated Scale[Row{i + 1}] = {scale[i]}")

                self._print(f"Matrix after Row{i + 1} operation")
                self._flush()

            self.pivot_positions.append((row, col))
            row += 1

        # === Final echelon form ===
        self.rank = row
        final = [
            "=" * 80,
            f"           UPPER TRIANGULAR (ROW ECHELON) FORM",
            f"                    RANK = {self.rank}",
            "=" * 80,
            "Final Echelon Matrix:"
        ]
        for i, r in enumerate(self.M):
            row_str = "  ".join(f"{self.round_sig(x):12.6g}" for x in r)
            final.append(f"R{i + 1}: {row_str}")
        final.append("")
        self._current = final
        self._flush()

    def get_result(self):
        return self.M, self.rank, self.pivot_positions