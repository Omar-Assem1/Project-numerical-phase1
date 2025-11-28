import math

from decimal import Context

class ForwardEliminator:
    """Performs Gaussian elimination with partial pivoting and records all steps"""

    def __init__(self, augmented_matrix, precision=4):
        self.n = len(augmented_matrix)
        self.M = [row[:] for row in augmented_matrix]
        self.rank = 0
        self.pivot_positions = []
        self.precision = precision

        # This will hold one big string per actual elimination step
        self.step_strings = []
        # Temporary buffer for the current step
        self._current_step = []

    def round_sig(self, x):
        # Create a context with the desired precision
        ctx = Context(prec=self.precision)
        # Normalize applies the precision to the number
        return float(ctx.create_decimal(x).normalize())

    def _flush_step(self):
        """Join the temporary buffer into one string and store it"""
        if self._current_step:
            self.step_strings.append("\n".join(self._current_step))
            self._current_step = []

    def _print(self, msg=""):
        """Print matrix to the current step buffer"""
        if msg:
            self._current_step.append(f"{msg}:")
        for i, r in enumerate(self.M):
            row_str = "  ".join(f"{self.round_sig(x):12.6g}" for x in r)
            self._current_step.append(f"R{i + 1}: {row_str}")
        self._current_step.append("")

    def eliminate(self):
        # Header (always step 0)
        header = [
            "=" * 75,
            "        FORWARD ELIMINATION WITH PARTIAL PIVOTING",
            "=" * 75,
            f"Precision: {self.precision} significant figures",
            "",
            "Initial Matrix:"
        ]
        for i, r in enumerate(self.M):
            row_str = "  ".join(f"{self.round_sig(x):12.6g}" for x in r)
            header.append(f"R{i + 1}: {row_str}")
        header.append("")
        self.step_strings.append("\n".join(header))

        row = 0
        step_count = 0

        for col in range(self.n):
            # Find pivot
            pivot_idx = None
            max_val = 0
            for i in range(row, self.n):
                val = abs(self.M[i][col])
                if val > 1e-10 and (pivot_idx is None or val > max_val):
                    max_val = val
                    pivot_idx = i

            if pivot_idx is None:
                continue

            # Swap rows
            if pivot_idx != row:
                self._current_step = []  # start new step
                self._current_step.append(f"SWAP: Row{row + 1} ↔ Row{pivot_idx + 1}")
                self._current_step.append("=" * 75)
                self.M[row], self.M[pivot_idx] = self.M[pivot_idx], self.M[row]
                self._print("After swap")
                self._flush_step()

            pivot = self.M[row][col]
            pivot_rounded = self.round_sig(pivot)

            # Eliminate below
            for i in range(row + 1, self.n):
                if abs(self.M[i][col]) < 1e-10:
                    continue

                step_count += 1
                self._current_step = []  # ← new step starts here

                numerator = self.M[i][col]
                denominator = self.M[row][col]
                factor = numerator / denominator
                factor_rounded = self.round_sig(factor)

                self._current_step.append(f"--- Step {step_count}: Row{i + 1} -= factor × Row{row + 1} ---")
                self._current_step.append(f"Factor = M[{i + 1}][{col + 1}] / M[{row + 1}][{col + 1}]")
                self._current_step.append(
                    f"       = {self.round_sig(numerator)} / {self.round_sig(denominator)}")
                self._current_step.append(f"       = {factor_rounded}")
                self._current_step.append(f"Operation: Row{i + 1} = Row{i + 1} - ({factor_rounded}) × Row{row + 1}")
                self._current_step.append("-" * 75)

                # Perform the actual elimination with rounded values
                for j in range(col, self.n + 1):
                    old_val = self.M[i][j]
                    pivot_row_val = self.M[row][j]
                    product = factor_rounded * pivot_row_val
                    new_val = old_val - product

                    old_val_r = self.round_sig(old_val )
                    pivot_row_val_r = self.round_sig(pivot_row_val )
                    product_r = self.round_sig(product )
                    new_val_r = self.round_sig(new_val )

                    if j == col:
                        self._current_step.append(
                            f"M[{i + 1}][{j + 1}]: {old_val_r} - ({factor_rounded} × {pivot_row_val_r}) = 0")
                        self.M[i][j] = 0.0
                    else:
                        self._current_step.append(
                            f"M[{i + 1}][{j + 1}]: {old_val_r} - ({factor_rounded} × {pivot_row_val_r}) = {new_val_r}")
                        self.M[i][j] = new_val_r

                self._print(f"Matrix after Row{i + 1} operation")
                self._flush_step()  # ← one complete step string is now stored

            self.pivot_positions.append((row, col))
            row += 1

        self.rank = row

        # Final echelon form (separate "step")
        final = [
            "=" * 75,
            f"           ROW ECHELON FORM  →  RANK = {self.rank}",
            "=" * 75,
            "Final Echelon Matrix:"
        ]
        for i, r in enumerate(self.M):
            row_str = "  ".join(f"{self.round_sig(x ):12.6g}" for x in r)
            final.append(f"R{i + 1}: {row_str}")
        final.append("")
        self.step_strings.append("\n".join(final))

    def get_result(self):
        return self.M, self.rank, self.pivot_positions