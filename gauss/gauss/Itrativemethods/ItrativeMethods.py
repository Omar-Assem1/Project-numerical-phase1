# iterative_methods.py
import math
from typing import List


class IterativeMethods:
    def __init__(self, n: int, A: List[List[float]], b: List[float],
                 X0: List[float], max_iter: int = 50, tol: float = 1e-6, precision: int = 6):
        self.n = n
        self.A = [row[:] for row in A]           # Coefficient matrix
        self.b = b[:]                            # Right-hand side
        self.X0 = X0[:]                          # Initial guess
        self.max_iter = max_iter
        self.tol = tol
        self.precision = precision

        # Internal state
        self.iterations = 0
        self.converged = False
        self.solution = None
        self.method_name = ""

    def round_sig(self, x: float, sig: int = None) -> float:
        if sig is None:
            sig = self.precision
        if abs(x) < 1e-20:
            return 0.0
        try:
            order = math.floor(math.log10(abs(x)))
            factor = 10 ** (order - sig + 1)
            return round(x / factor) * factor
        except (OverflowError, ValueError):
            return x

    def _round_vec(self, vec: List[float]) -> List[float]:
        return [self.round_sig(v, self.precision) for v in vec]

    def _print_vec(self, name: str, vec: List[float], steps: List[str]):
        rounded = self._round_vec(vec)
        steps.append(f"{name}: [{'  '.join(f'{v:12.6g}' for v in rounded)}]")

    def _print_iteration_header(self, method: str, steps: List[str]):
        self.method_name = method.upper()
        steps.append("\n" + "="*85)
        steps.append(f"        {self.method_name} ITERATIVE METHOD")
        steps.append("="*85)
        steps.append(f"Precision: {self.precision} significant figures")
        steps.append(f"Tolerance: {self.tol:.2e} | Max iterations: {self.max_iter}")
        steps.append("Initial guess X⁽⁰⁾:")
        self._print_vec("X⁽⁰⁾", self.X0, steps)
        steps.append("")

        # Show formulas
        steps.append("Iteration formulas:")
        for i in range(self.n):
            terms = [f"{self.round_sig(self.b[i], self.precision):.6g}"]
            for j in range(self.n):
                if i == j:
                    continue
                coef = self.round_sig(-self.A[i][j] / self.A[i][i], self.precision)
                sign = " + " if coef >= 0 else " - "
                val = abs(coef)
                var_type = "new" if method == "seidel" and j < i else "old"
                terms.append(f"{sign}{val:.6g}·x{j+1}({var_type})")
            formula = f"x{i+1}(k+1) = (1/{self.round_sig(self.A[i][i], self.precision):.6g}) × ({' '.join(terms)})"
            steps.append(f"   {formula}")
        steps.append("")

    def _check_diagonal_dominance(self, steps: List[str]):
        steps.append("Diagonal dominance check:")
        dominant = True
        for i in range(self.n):
            diag = abs(self.A[i][i])
            off = sum(abs(self.A[i][j]) for j in range(self.n) if j != i)
            status = "OK" if diag >= off else "Not strictly dominant"
            if diag < off:
                dominant = False
            steps.append(f"  Row {i+1}: |a{i+1}{i+1}| = {diag:.6g} ≥ Σ|others| = {off:.6g} → {status}")
        if dominant:
            steps.append("→ Matrix is strictly diagonally dominant → convergence guaranteed")
        else:
            steps.append("→ Not strictly dominant → convergence not guaranteed")
        steps.append("")

    def jacobi(self, steps: List[str]) -> List[float]:
        self._print_iteration_header("jacobi", steps)
        self._check_diagonal_dominance(steps)

        X = self.X0[:]
        X_new = [0.0] * self.n
        steps.append("Starting Jacobi iterations...\n")
        steps.append(f"{'Iter':>4}  | {'x₁':>12} {'x₂':>12} {'x₃':>12}  | {'Max Error':>12}")
        steps.append("-" * 70)

        for k in range(1, self.max_iter + 1):
            self.iterations = k
            X_old = X[:]

            for i in range(self.n):
                sigma = sum(self.round_sig(self.A[i][j] * X[j], self.precision)
                           for j in range(self.n) if j != i)
                val = self.b[i] - sigma
                X_new[i] = self.round_sig(val / self.A[i][i], self.precision)

            X = X_new[:]
            errors = [abs(X[i] - X_old[i]) for i in range(self.n)]
            max_err = max(errors) if errors else 0

            # Display rounded values
            X_disp = self._round_vec(X)
            err_disp = self.round_sig(max_err, self.precision)
            steps.append(f"{k:>4}  | " + " ".join(f"{x:12.6g}" for x in X_disp) +
                         f"  | {err_disp:12.6g}")

            if max_err < self.tol:
                self.converged = True
                break

        steps.append("")
        if self.converged:
            steps.append("CONVERGED!")
        else:
            steps.append("Maximum iterations reached (did NOT converge)")

        steps.append("\n" + "="*85)
        steps.append(f"        JACOBI FINAL RESULT")
        steps.append("="*85)
        self.solution = self._round_vec(X)
        for i, val in enumerate(self.solution, 1):
            steps.append(f"x{i} = {val:.10g}")
        steps.append(f"After {self.iterations} iteration(s)")
        steps.append("")

        return self.solution

    def gauss_seidel(self, steps: List[str]) -> List[float]:
        self._print_iteration_header("seidel", steps)
        self._check_diagonal_dominance(steps)

        X = self.X0[:]
        steps.append("Starting Gauss-Seidel iterations...\n")
        steps.append(f"{'Iter':>4}  | {'x₁':>12} {'x₂':>12} {'x₃':>12}  | {'Max Error':>12}")
        steps.append("-" * 70)

        for k in range(1, self.max_iter + 1):
            self.iterations = k
            X_old = X[:]

            for i in range(self.n):
                sigma = sum(self.round_sig(self.A[i][j] * X[j], self.precision)
                           for j in range(self.n) if j != i)
                val = self.b[i] - sigma
                X[i] = self.round_sig(val / self.A[i][i], self.precision)

            errors = [abs(X[i] - X_old[i]) for i in range(self.n)]
            max_err = max(errors) if errors else 0

            X_disp = self._round_vec(X)
            err_disp = self.round_sig(max_err, self.precision)
            steps.append(f"{k:>4}  | " + " ".join(f"{x:12.6g}" for x in X_disp) +
                         f"  | {err_disp:12.6g}")

            if max_err < self.tol:
                self.converged = True
                break

        steps.append("")
        if self.converged:
            steps.append("CONVERGED!")
        else:
            steps.append("Maximum iterations reached (did NOT converge)")

        steps.append("\n" + "="*85)
        steps.append(f"        GAUSS-SEIDEL FINAL RESULT")
        steps.append("="*85)
        self.solution = self._round_vec(X)
        for i, val in enumerate(self.solution, 1):
            steps.append(f"x{i} = {val:.10g}")
        steps.append(f"After {self.iterations} iteration(s)")
        steps.append("")

        return self.solution


# ———————————————————————— EXAMPLE USAGE ————————————————————————
if __name__ == "__main__":
    A = [
        [10, -1, 2, 0],
        [-1, 11, -1, 3],
        [2, -1, 10, -1],
        [0, 3, -1, 8]
    ]

    b = [6, 25, -11, 15]
    X0 = [0, 0, 0,0]

    steps = []

    print("JACOBI METHOD")
    solver = IterativeMethods(n=4, A=A, b=b, X0=X0, max_iter=100, tol=1e-5, precision=6)
    solver.jacobi(steps)
    print("\n".join(steps))

    steps.clear()
    print("\n" + "="*90)
    print("GAUSS-SEIDEL METHOD")
    solver = IterativeMethods(n=4, A=A, b=b, X0=X0, max_iter=100, tol=1e-5, precision=6)
    solver.gauss_seidel(steps)
    print("\n".join(steps))