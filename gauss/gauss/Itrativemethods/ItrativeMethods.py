# Itrativemethods/IterativeMethods.py
import math
from typing import List, Tuple, Optional


class IterativeMethods:
    def __init__(self, n: int, A: List[List[float]], b: List[float],
                 X0: List[float], max_iter: int = 50, tol: float = 1e-6, precision: int = 6):
        self.n = n
        self.A = [row[:] for row in A]
        self.b = b[:]
        self.X0 = X0[:]
        self.max_iter = max_iter
        self.tol = tol
        self.precision = precision

        # Step tracking
        self.step_strings = []
        self._current = []

    def round_sig(self, x: float, sig: int = None) -> float:
        if sig is None:
            sig = self.precision
        if abs(x) < 1e-20:
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

    def _print_vec(self, name: str, vec: List[float]):
        rounded = [self.round_sig(v) for v in vec]
        self._current.append(f"{name}: [{'  '.join(f'{v:12.6g}' for v in rounded)}]")

    def _print_table_header(self):
        header = f"{'Iter':>4}  | " + " ".join(f"x{i+1:>12}" for i in range(self.n)) + f"  | {'Max Error':>12}"
        self._current.append(header)
        self._current.append("-" * (6 + 15 * self.n + 15))

    def _add_iteration_row(self, k: int, X: List[float], error: float):
        X_disp = [self.round_sig(x) for x in X]
        err_disp = self.round_sig(error) if error > 0 else 0.0
        row = f"{k:>4}  | " + " ".join(f"{x:12.6g}" for x in X_disp) + f"  | {err_disp:12.6g}"
        self._current.append(row)

    def _check_diagonal_dominance(self):
        self._current.append("Diagonal dominance check:")
        dominant = True
        for i in range(self.n):
            diag = abs(self.A[i][i])
            off = sum(abs(self.A[i][j]) for j in range(self.n) if j != i)
            status = "STRICT" if diag > off else "WEAK" if diag >= off else "NO"
            if diag <= off:
                dominant = False
            self._current.append(f"  Row {i+1}: |a{i+1}{i+1}| = {diag:.6g} vs Σ|others| = {off:.6g} → {status}")
        self._current.append("→ Strictly diagonally dominant → convergence guaranteed!" if dominant else
                             "→ Not strictly dominant → convergence not guaranteed")
        self._current.append("")

    def jacobi(self) -> Tuple[List[float], int]:
        self.step_strings = []
        self._current = [
            "=" * 85,
            "                 JACOBI ITERATIVE METHOD",
            "=" * 85,
            f"Precision: {self.precision} sig. figs | Tolerance: {self.tol:.2e} | Max iter: {self.max_iter}",
            "",
            "Initial guess:",
        ]
        self._print_vec("X⁽⁰⁾", self.X0)
        self._check_diagonal_dominance()
        self._current.append("Starting Jacobi iterations...")
        self._print_table_header()
        self._flush()

        X_old = self.X0[:]
        X_new = [0.0] * self.n

        for k in range(1, self.max_iter + 1):
            for i in range(self.n):
                sigma = sum(self.A[i][j] * X_old[j] for j in range(self.n) if j != i)
                X_new[i] = (self.b[i] - sigma) / self.A[i][i]
                X_new[i] = self.round_sig(X_new[i])

            error = max(abs(X_new[i] - X_old[i]) for i in range(self.n))
            self._add_iteration_row(k, X_new, error)

            if error < self.tol:
                self._current = [
                    "CONVERGED!",
                    "",
                    "=" * 85,
                    f" JACOBI CONVERGED IN {k} ITERATIONS",
                    "=" * 85,
                ]
                self._print_vec("Final solution", X_new)
                for i, val in enumerate(X_new, 1):
                    self._current.append(f"x{i} = {self.round_sig(val):.10g}")
                self._current.append("")
                self._flush()
                return X_new, k

            self._flush()
            X_old = X_new[:]

        # Not converged
        self._current = [
            "DID NOT CONVERGE within max iterations",
            "",
            "=" * 85,
            f" JACOBI – LAST APPROXIMATION (after {self.max_iter} iter)",
            "=" * 85,
        ]
        self._print_vec("Last X", X_new)
        self._flush()
        return X_new, self.max_iter

    def gauss_seidel(self) -> Tuple[List[float], int]:
        self.step_strings = []
        self._current = [
            "=" * 85,
            "              GAUSS-SEIDEL ITERATIVE METHOD",
            "=" * 85,
            f"Precision: {self.precision} sig. figs | Tolerance: {self.tol:.2e} | Max iter: {self.max_iter}",
            "",
            "Initial guess:",
        ]
        self._print_vec("X⁽⁰⁾", self.X0)
        self._check_diagonal_dominance()
        self._current.append("Starting Gauss-Seidel iterations...")
        self._print_table_header()
        self._flush()

        X = self.X0[:]

        for k in range(1, self.max_iter + 1):
            X_old = X[:]

            for i in range(self.n):
                sigma = sum(self.A[i][j] * X[j] for j in range(i)) + \
                        sum(self.A[i][j] * X_old[j] for j in range(i + 1, self.n))
                X[i] = (self.b[i] - sigma) / self.A[i][i]
                X[i] = self.round_sig(X[i])

            error = max(abs(X[i] - X_old[i]) for i in range(self.n))
            self._add_iteration_row(k, X, error)

            if error < self.tol:
                self._current = [
                    "CONVERGED!",
                    "",
                    "=" * 85,
                    f" GAUSS-SEIDEL CONVERGED IN {k} ITERATIONS",
                    "=" * 85,
                ]
                self._print_vec("Final solution", X)
                for i, val in enumerate(X, 1):
                    self._current.append(f"x{i} = {self.round_sig(val):.10g}")
                self._current.append("")
                self._flush()
                return X, k

            self._flush()

        self._current = [
            "DID NOT CONVERGE within max iterations",
            "",
            "=" * 85,
            f" GAUSS-SEIDEL – LAST APPROXIMATION (after {self.max_iter} iter)",
            "=" * 85,
        ]
        self._print_vec("Last X", X)
        self._flush()
        return X, self.max_iter