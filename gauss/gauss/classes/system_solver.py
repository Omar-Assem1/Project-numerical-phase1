import math
class SystemSolver:
    """Back substitution with full step-by-step recording"""
    def __init__(self, echelon_matrix, rank, n, pivot_positions, precision=4):
        self.M = echelon_matrix
        self.rank = rank
        self.n = n
        self.pivot_positions = pivot_positions
        self.precision = precision

    def round_sig(self, x, sig):
        if x == 0:
            return 0.0
        order = math.floor(math.log10(abs(x)))
        factor = 10 ** (order - sig + 1)
        return round(x / factor) * factor

    def solve(self, steps):
        steps.append("\n" + "=" * 75)
        steps.append("                    BACK SUBSTITUTION")
        steps.append("=" * 75)
        steps.append(f"Precision: {self.precision} significant figures\n")

        # Check inconsistency
        for i in range(self.rank, self.n):
            if abs(self.M[i][self.n]) > 1e-8:
                steps.append("\n" + "=" * 75)
                steps.append("NO SOLUTION – System is inconsistent")
                steps.append("=" * 75)
                steps.append(f"Row {i + 1} → 0 = {self.round_sig(self.M[i][self.n], self.precision)}")
                return None

        if self.rank < self.n:
            steps.append("\n" + "=" * 75)
            steps.append(f"INFINITE SOLUTIONS ({self.n - self.rank} free variable(s))")
            steps.append("=" * 75)
            pivot_cols = {col for _, col in self.pivot_positions}
            free = [j + 1 for j in range(self.n) if j not in pivot_cols]
            steps.append("Free variables: " + ", ".join(f"x{v}" for v in free))
            return None

        steps.append("Unique solution exists. Performing backward substitution:\n")
        x = [0.0] * self.n

        for i in range(self.n - 1, -1, -1):
            row = self.M[i]

            steps.append(f"{'=' * 75}")
            steps.append(f"STEP {self.n - i}: Solving for x{i + 1}")
            steps.append(f"{'=' * 75}")

            # Equation
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
            steps.append(f"\nFrom Row {i + 1}:")
            steps.append(f"  {' '.join(eq_parts)} = {rhs}")

            # Known sum
            known_sum = 0.0
            sum_details = []
            for j in range(i + 1, self.n):
                coef = self.round_sig(row[j], self.precision)
                if abs(coef) > 1e-10:
                    term = coef * x[j]
                    term_r = self.round_sig(term, self.precision)
                    sum_details.append(f"({coef} × {self.round_sig(x[j], self.precision)})")
                    known_sum += term_r
            known_sum_r = self.round_sig(known_sum, self.precision)

            diagonal = self.round_sig(row[i], self.precision)

            if sum_details:
                steps.append(f"\nSubstitute known values:")
                steps.append(f"  Sum of known terms = {' + '.join(sum_details)}")
                steps.append(f"                     = {known_sum_r}")
                steps.append(f"\nIsolate x{i + 1}:")
                steps.append(f"  {diagonal}·x{i + 1} = {rhs} - {known_sum_r}")
                numerator_r = self.round_sig(rhs - known_sum_r, self.precision)
                steps.append(f"  {diagonal}·x{i + 1} = {numerator_r}")
            else:
                steps.append(f"\nIsolate x{i + 1}:")
                steps.append(f"  {diagonal}·x{i + 1} = {rhs}")
                numerator_r = rhs

            result = numerator_r / diagonal
            result_r = self.round_sig(result, self.precision)
            steps.append(f"  x{i + 1} = {numerator_r} / {diagonal}")
            steps.append(f"  x{i + 1} = {result_r}\n")

            x[i] = result_r

        steps.append("\n" + "=" * 75)
        steps.append("                    FINAL SOLUTION")
        steps.append("=" * 75)
        for i, val in enumerate(x, 1):
            steps.append(f"x{i} = {self.round_sig(val, self.precision)}")
        steps.append("")

        return x