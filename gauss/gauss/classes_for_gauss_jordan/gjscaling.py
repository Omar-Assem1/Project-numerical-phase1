# classes_for_gauss_jordan/gjscaling.py
import math


class GaussJordanEliminatorScaling:
    def __init__(self, augmented_matrix, precision=6):
        self.n = len(augmented_matrix)
        self.M = [row[:] for row in augmented_matrix]
        self.rank = 0
        self.pivot_positions = []
        self.precision = precision

        self.step_strings = []
        self._current = []

    def round_sig(self, x, sig=None):
        if sig is None:
            sig = self.precision
        if x == 0:
            return 0.0
        try:
            order = math.floor(math.log10(abs(x)))
            factor = 10 ** (order - sig + 1)
            return round(x / factor) * factor
        except:
            return x

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
        # Step 0: Header + Scales
        header = [
            "=" * 80,
            "    GAUSS-JORDAN WITH SCALED PARTIAL PIVOTING → RREF",
            "=" * 80,
            f"Precision: {self.precision} significant figures",
            "",
            "Initial Augmented Matrix:"
        ]
        self._current = header
        self._print()

        scale = [0.0] * self.n
        self._current.append("-" * 80)
        self._current.append("COMPUTING SCALE FACTORS")
        self._current.append("-" * 80)
        for i in range(self.n):
            row_max = max(abs(self.M[i][j]) for j in range(self.n))
            scale[i] = self.round_sig(row_max if row_max > 0 else 1.0)
            self._current.append(f"Scale[Row{i + 1}] = {scale[i]:.6g}")
        self._current.append("")
        self._flush()

        h = 0
        for col in range(self.n):
            if h >= self.n:
                break

            # Find best scaled pivot
            best_idx = h
            best_ratio = -1.0
            self._current = [f"SEARCHING PIVOT FOR COLUMN {col + 1}"]
            self._current.append("-" * 80)

            for i in range(h, self.n):
                ratio = abs(self.M[i][col]) / scale[i] if scale[i] > 0 else 0.0
                ratio_r = self.round_sig(ratio)
                marker = " ← BEST" if ratio > best_ratio + 1e-12 else ""
                if marker:
                    best_ratio = ratio
                    best_idx = i
                self._current.append(f"  Row{i + 1}: |{self.round_sig(self.M[i][col])}| / {scale[i]:.6g} = {ratio_r}{marker}")

            if best_ratio < 1e-12:
                self._current.append("No significant pivot. Skipping column.")
                self._flush()
                continue

            # Swap
            if best_idx != h:
                self._current.append(f"\nSWAP Row{h + 1} ↔ Row{best_idx + 1}")
                self.M[h], self.M[best_idx] = self.M[best_idx], self.M[h]
                scale[h], scale[best_idx] = scale[best_idx], scale[h]
                self._print("After swap")
                self._flush()

            # Normalize pivot row
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

            # Eliminate all other rows
            for i in range(self.n):
                if i == h or abs(self.M[i][col]) < 1e-10:
                    continue

                factor = self.M[i][col]
                factor_r = self.round_sig(factor)

                self._current = [
                    f"ELIMINATE IN ROW {i + 1}",
                    f"Row{i + 1} -= ({factor_r}) × Row{h + 1}",
                    "-" * 80
                ]

                for j in range(self.n + 1):
                    self.M[i][j] = self.round_sig(self.M[i][j] - factor * self.M[h][j])
                self.M[i][col] = 0.0

                # Update scale
                new_max = max(abs(self.M[i][j]) for j in range(self.n))
                if new_max > 0:
                    scale[i] = self.round_sig(new_max)
                    self._current.append(f"Updated Scale[Row{i + 1}] = {scale[i]:.6g}")

                self._print(f"After eliminating in Row{i + 1}")
                self._flush()

            self.pivot_positions.append((h, col))
            h += 1

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