# classes_for_gauss_jordan/gjscaling.py
import math


class GaussJordanEliminatorScaling:
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
        steps.append("    GAUSS-JORDAN WITH SCALED PARTIAL PIVOTING → RREF")
        steps.append("=" * 80)
        steps.append(f"Precision: {self.precision} significant figures")
        self._print("Initial Augmented Matrix", steps)

        # Scale factors
        scale = [0.0] * self.n
        steps.append("\n" + "-" * 80)
        steps.append("COMPUTING SCALE FACTORS")
        steps.append("-" * 80)
        for i in range(self.n):
            row_max = max(abs(self.M[i][j]) for j in range(self.n))
            scale[i] = self.round_sig(row_max if row_max > 0 else 1.0)
            steps.append(f"Scale[Row{i + 1}] = {scale[i]:.6g}")
        steps.append("")

        h = 0
        for col in range(self.n):
            if h >= self.n:
                break

            steps.append(f"\nSearching pivot for column {col + 1}:")
            best_idx = None
            best_ratio = -1.0

            for i in range(h, self.n):
                ratio = self.round_sig(abs(self.M[i][col]) / scale[i]) if scale[i] > 0 else 0.0
                marker = " ← BEST" if ratio > best_ratio + 1e-12 else ""
                if marker:
                    best_ratio = ratio
                    best_idx = i
                steps.append(f"  Row{i + 1}: ratio = {ratio:.6g}{marker}")

            if best_idx is None or best_ratio < 1e-12:
                steps.append(f"No significant pivot in column {col + 1}. Skipping.")
                continue

            if best_idx != h:
                steps.append(f"\nSwap Row{h + 1} ↔ Row{best_idx + 1}")
                self.M[h], self.M[best_idx] = self.M[best_idx], self.M[h]
                scale[h], scale[best_idx] = scale[best_idx], scale[h]
                self._print("After swap", steps)

            pivot = self.M[h][col]
            pivot_r = self.round_sig(pivot)
            steps.append(f"\nRow{h + 1} ÷ {pivot_r}  →  make pivot = 1")
            for j in range(self.n + 1):
                self.M[h][j] = self.round_sig(self.M[h][j] / pivot)
            self._print("After normalizing pivot row", steps)

            for i in range(self.n):
                if i == h or abs(self.M[i][col]) < 1e-10:
                    continue

                factor = self.round_sig(self.M[i][col])
                factor_r = self.round_sig(factor)
                steps.append(f"Row{i + 1} -= ({factor_r}) × Row{h + 1}")

                for j in range(self.n + 1):
                    product = self.round_sig(factor * self.M[h][j])
                    self.M[i][j] = self.round_sig(self.M[i][j] - product)

                self.M[i][col] = 0.0
                new_max = max(abs(self.M[i][j]) for j in range(self.n))
                if new_max > 0:
                    scale[i] = self.round_sig(new_max)
                    steps.append(f"Updated Scale[Row{i + 1}] = {scale[i]:.6g}")

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
        return self.M, self.rank, self.pivot_positions# classes_for_gauss_jordan/gjscaling.py
import math


class GaussJordanEliminatorScaling:
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
        steps.append("    GAUSS-JORDAN WITH SCALED PARTIAL PIVOTING → RREF")
        steps.append("=" * 80)
        steps.append(f"Precision: {self.precision} significant figures")
        self._print("Initial Augmented Matrix", steps)

        # Scale factors
        scale = [0.0] * self.n
        steps.append("\n" + "-" * 80)
        steps.append("COMPUTING SCALE FACTORS")
        steps.append("-" * 80)
        for i in range(self.n):
            row_max = max(abs(self.M[i][j]) for j in range(self.n))
            scale[i] = self.round_sig(row_max if row_max > 0 else 1.0)
            steps.append(f"Scale[Row{i + 1}] = {scale[i]:.6g}")
        steps.append("")

        h = 0
        for col in range(self.n):
            if h >= self.n:
                break

            steps.append(f"\nSearching pivot for column {col + 1}:")
            best_idx = None
            best_ratio = -1.0

            for i in range(h, self.n):
                ratio = self.round_sig(abs(self.M[i][col]) / scale[i]) if scale[i] > 0 else 0.0
                marker = " ← BEST" if ratio > best_ratio + 1e-12 else ""
                if marker:
                    best_ratio = ratio
                    best_idx = i
                steps.append(f"  Row{i + 1}: ratio = {ratio:.6g}{marker}")

            if best_idx is None or best_ratio < 1e-12:
                steps.append(f"No significant pivot in column {col + 1}. Skipping.")
                continue

            if best_idx != h:
                steps.append(f"\nSwap Row{h + 1} ↔ Row{best_idx + 1}")
                self.M[h], self.M[best_idx] = self.M[best_idx], self.M[h]
                scale[h], scale[best_idx] = scale[best_idx], scale[h]
                self._print("After swap", steps)

            pivot = self.M[h][col]
            pivot_r = self.round_sig(pivot)
            steps.append(f"\nRow{h + 1} ÷ {pivot_r}  →  make pivot = 1")
            for j in range(self.n + 1):
                self.M[h][j] = self.round_sig(self.M[h][j] / pivot)
            self._print("After normalizing pivot row", steps)

            for i in range(self.n):
                if i == h or abs(self.M[i][col]) < 1e-10:
                    continue

                factor = self.round_sig(self.M[i][col])
                factor_r = self.round_sig(factor)
                steps.append(f"Row{i + 1} -= ({factor_r}) × Row{h + 1}")

                for j in range(self.n + 1):
                    product = self.round_sig(factor * self.M[h][j])
                    self.M[i][j] = self.round_sig(self.M[i][j] - product)

                self.M[i][col] = 0.0
                new_max = max(abs(self.M[i][j]) for j in range(self.n))
                if new_max > 0:
                    scale[i] = self.round_sig(new_max)
                    steps.append(f"Updated Scale[Row{i + 1}] = {scale[i]:.6g}")

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