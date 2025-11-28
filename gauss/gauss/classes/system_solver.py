import math

class SystemSolver:
    """Back substitution — one big string per variable solved"""
    def __init__(self, echelon_matrix, rank, n, pivot_positions, precision=4):
        self.M = echelon_matrix
        self.rank = rank
        self.n = n
        self.pivot_positions = pivot_positions
        self.precision = precision

        self.step_strings = []
        self._current = []

    def round_sig(self, x, sig):
        if x == 0:
            return 0.0
        order = math.floor(math.log10(abs(x)))
        factor = 10 ** (order - sig + 1)
        return round(x / factor) * factor

    def _flush(self):
        if self._current:
            self.step_strings.append("\n".join(self._current))
            self._current = []

    def solve(self):
        # === Header ===
        self._current = [
            "=" * 75,
            "                    BACK SUBSTITUTION",
            "=" * 75,
            f"Precision: {self.precision} significant figures",
            ""
        ]
        self._flush()

        # Inconsistency check
        for i in range(self.rank, self.n):
            if abs(self.M[i][self.n]) > 1e-8:
                self._current = [
                    "=" * 75,
                    "NO SOLUTION – System is inconsistent",
                    "=" * 75,
                    f"Row {i + 1} → 0 = {self.round_sig(self.M[i][self.n], self.precision)}"
                ]
                self._flush()
                return None

        if self.rank < self.n:
            pivot_cols = {col for _, col in self.pivot_positions}
            free = [j + 1 for j in range(self.n) if j not in pivot_cols]
            self._current = [
                "=" * 75,
                f"INFINITE SOLUTIONS ({self.n - self.rank} free variable(s))",
                "=" * 75,
                "Free variables: " + ", ".join(f"x{v}" for v in free)
            ]
            self._flush()
            return None

        self._current = ["Unique solution exists. Performing backward substitution:", ""]
        self._flush()

        x = [0.0] * self.n

        for i in range(self.n - 1, -1, -1):
            self._current = []
            row = self.M[i]

            self._current.extend([
                f"{'=' * 75}",
                f"STEP {self.n - i}: Solving for x{i + 1}",
                f"{'=' * 75}",
                ""
            ])

            # Original equation
            eq_parts = []
            for j in range(self.n):
                coef = self.round_sig(row[j], self.precision)
                if abs(coef) > 1e-10:
                    if not eq_parts:
                        eq_parts.append(f"{coef}·x{j + 1}")
                    else:
                        sign = "+" if coef >= 0 else "-"
                        eq_parts.append(f" {sign} {abs(coef)}·x{j + 1}")
            rhs = self.round_sig(row[self.n], self.precision)
            self._current.append(f"From Row {i + 1}:")
            self._current.append(f"  {' '.join(eq_parts)} = {rhs}")
            self._current.append("")

            # Known terms
            known_sum = 0.0
            sum_details = []
            for j in range(i + 1, self.n):
                coef = self.round_sig(row[j], self.precision)
                if abs(coef) > 1e-10:
                    term = coef * x[j]
                    term_r = self.round_sig(term, self.precision)
                    sum_details.append(f"({coef} × {self.round_sig(x[j], self.precision)}) = {term_r}")
                    known_sum += term_r

            diagonal = self.round_sig(row[i], self.precision)

            if sum_details:
                self._current.append("Substitute known values:")
                self._current.append("  Sum of known terms = " + " + ".join(sum_details))
                self._current.append(f"                     = {self.round_sig(known_sum, self.precision)}")
                self._current.append("")
                self._current.append(f"Isolate x{i + 1}:")
                self._current.append(f"  {diagonal}·x{i + 1} = {rhs} − {self.round_sig(known_sum, self.precision)}")
                numerator_r = self.round_sig(rhs + (-known_sum), self.precision)
                self._current.append(f"  {diagonal}·x{i + 1} = {numerator_r}")
            else:
                self._current.append(f"Isolate x{i + 1}:")
                self._current.append(f"  {diagonal}·x{i + 1} = {rhs}")
                numerator_r = rhs

            result = numerator_r / diagonal
            result_r = self.round_sig(result, self.precision)

            self._current.extend([
                f"  x{i + 1} = {numerator_r} / {diagonal}",
                f"  x{i + 1} = {result_r}",
                ""
            ])

            x[i] = result_r
            self._flush()

        # === Final solution ===
        self._current = [
            "=" * 75,
            "                    FINAL SOLUTION",
            "=" * 75
        ]
        for i, val in enumerate(x, 1):
            self._current.append(f"x{i} = {self.round_sig(val, self.precision)}")
        self._current.append("")
        self._flush()

        return x