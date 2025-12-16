import math
import time
import sympy as sp
from decimal import Decimal, Context


class FixedPointMethod:

    def __init__(self, g_equation_str, initial_guess,
                 epsilon=0.00001, max_iterations=50, significant_figures=5):

        self.g_equation_str = g_equation_str
        self.x0 = initial_guess
        self.epsilon = epsilon * 100  # Convert to percentage to match relative error calculation
        self.max_iterations = max_iterations
        self.significant_figures = significant_figures

        # Parse g equation
        self.x = sp.Symbol('x')
        try:
            self.g = sp.sympify(g_equation_str)
            self.g_prime = sp.diff(self.g, self.x)
        except Exception as e:
            raise ValueError(f"Error parsing g equation: {e}")

        # Results storage
        self.root = None
        self.iterations = 0
        self.relative_error = None
        self.execution_time = 0
        self.iteration_history = []
        self.converged = False
        self.error_message = None
        self.step_strings = []
        self.last_significant_figures = 0



    def evaluate_function(self, func, x_val):
        """Safely evaluate a symbolic function at x_val."""
        try:
            return float(func.subs(self.x, x_val))
        except Exception as e:
            raise ValueError(f"Error evaluating function at x={x_val}: {e}")

    def round_sig(self, x):
        """Round number to specified significant figures."""
        if x == 0:
            return 0.0
        try:
            x_str = str(x)
            ctx = Context(prec=self.significant_figures)
            return float(ctx.create_decimal(x_str).normalize())
        except:
            return float(x)

    def calculate_relative_error(self, x_new, x_old):
        """Calculate approximate relative error."""
        if abs(x_new) < 1e-15:
            abs_error = abs(x_new - x_old)
            return 0.0 if abs_error < 1e-15 else float('inf')
        return abs((x_new - x_old) / x_new) * 100

    def count_significant_figures(self, x_new, x_old):
        """Count correct significant figures."""
        if x_new == 0:
            return 0
        rel_error = abs((x_new - x_old) / x_new)
        if rel_error == 0:
            return 15
        if rel_error < 1e-10:
            return 10
        n = 2 - math.log10(2 * rel_error)
        return max(0, int(n))

    def check_convergence_condition(self):
        """Check if |g'(x)| < 1 at initial guess."""
        try:
            g_prime_val = self.evaluate_function(self.g_prime, self.x0)
            return abs(g_prime_val) < 1, g_prime_val
        except:
            return False, None

    def solve(self, show_steps=False):
        """Solve using Fixed Point Iteration."""
        start_time = time.time()
        self.step_strings = []

        # Header
        header = [
            "=" * 70,
            "Fixed Point Iteration Method",
            "=" * 70,
            f"Iteration form: x = g(x) = {self.g_equation_str}",
            f"Initial guess: x₀ = {self.x0}",
            f"Tolerance (ε): {self.epsilon/100} ({self.epsilon}%)",
            f"Max iterations: {self.max_iterations}",
            "=" * 70
        ]
        self.step_strings.append("\n".join(header))

        # Convergence check
        converges, g_prime_val = self.check_convergence_condition()
        if g_prime_val is not None and not converges:
            warning = (
                f"⚠ Warning: |g'(x₀)| = {abs(g_prime_val):.6f} >= 1. "
                "Method may diverge!"
            )
            self.step_strings.append(warning)

        x_old = self.x0
        x_new = self.x0
        self.iteration_history = []
        last_valid_x = self.x0

        for i in range(self.max_iterations):
            try:
                x_new_raw = self.evaluate_function(self.g, x_old)

                if not math.isfinite(x_new_raw):
                    self.error_message = f"Iteration {i + 1}: Non-finite value (NaN/Inf)"
                    self.step_strings.append(self.error_message)
                    x_new = last_valid_x
                    break

                x_new = self.round_sig(x_new_raw)
                last_valid_x = x_new

                rel_error = self.calculate_relative_error(x_new, x_old)
                rel_error = self.round_sig(rel_error) if math.isfinite(rel_error) else rel_error

                g_val = self.round_sig(x_new_raw) if math.isfinite(x_new_raw) else x_new_raw

                # Calculate significant figures with exact solution check
                if rel_error == 0:
                    sig_figs = self.significant_figures  # Use precision when exact solution found
                else:
                    sig_figs = self.count_significant_figures(x_new, x_old)
                
                self.last_significant_figures = sig_figs  # Store the last computed significant figures

                x_old_rounded = self.round_sig(x_old)
                x_new_rounded = self.round_sig(x_new)

                iteration_data = {
                    'iteration': i + 1,
                    'x_old': x_old_rounded,
                    'x_new': x_new_rounded,
                    'g(x_old)': g_val,
                    'relative_error': rel_error,
                    'significant_figures': sig_figs
                }
                self.iteration_history.append(iteration_data)

                iteration_step = [
                    f"Iteration {i + 1}:",
                    f"  x_old = {x_old_rounded}",
                    f"  x_new = g(x_old) = {x_new_rounded}",
                    f"  g(x_old) = {g_val}" if math.isfinite(g_val) else "  g(x_old) = undefined",
                    f"  |εₐ| = {rel_error:.6f}%" if math.isfinite(rel_error) else "  |εₐ| = N/A",
                    f"  Significant figures: {sig_figs}"
                ]
                self.step_strings.append("\n".join(iteration_step))

                if show_steps:
                    print("\n".join(iteration_step))

                # Convergence checks - both conditions must be satisfied for strict epsilon accuracy
                epsilon_satisfied = math.isfinite(rel_error) and rel_error <= self.epsilon
                precision_satisfied = self.significant_figures and sig_figs >= self.significant_figures
                
                if epsilon_satisfied and precision_satisfied:
                    self.converged = True
                    self.root = x_new_rounded
                    self.iterations = i + 1
                    self.relative_error = rel_error
                    self.step_strings.append(f"✓ Converged! Error {rel_error:.6f}% <= {self.epsilon} AND {sig_figs} >= {self.significant_figures} sig figs")
                    break

                # Check for stagnation (relative error not improving and good precision achieved)
                if i >= 10 and sig_figs >= self.significant_figures:
                    # Check if relative error has been constant for several iterations
                    if len(self.iteration_history) >= 5:
                        last_5_errors = [self.iteration_history[-j]['relative_error'] for j in range(1, 6)]
                        if all(abs(err - rel_error) < 1e-10 for err in last_5_errors):
                            self.converged = True
                            self.root = x_new_rounded
                            self.iterations = i + 1
                            self.relative_error = rel_error
                            self.step_strings.append(f"✓ Converged! Numerical precision limit reached with {sig_figs} >= {self.significant_figures} sig figs")
                            break

                if i > 5 and abs(x_new) > 1e10:
                    self.error_message = "Method diverging (values too large)"
                    self.step_strings.append(self.error_message)
                    self.root = self.round_sig(last_valid_x)
                    self.iterations = i + 1
                    self.relative_error = rel_error if math.isfinite(rel_error) else None
                    break

                x_old = x_new

            except Exception as e:
                self.error_message = f"Error at iteration {i + 1}: {str(e)}"
                self.step_strings.append(self.error_message)
                self.root = self.round_sig(last_valid_x)
                self.iterations = i + 1
                self.relative_error = None
                break

        if self.root is None:
            self.root = self.round_sig(last_valid_x)

        if not self.converged and not self.error_message:
            self.iterations = self.max_iterations
            self.relative_error = self.round_sig(rel_error) if 'rel_error' in locals() and math.isfinite(
                rel_error) else None
            self.error_message = f"No convergence in {self.max_iterations} iterations"
            self.step_strings.append(self.error_message)

        self.execution_time = self.round_sig(time.time() - start_time)

        # Final results
        results = ["", "=" * 70, "RESULTS", "=" * 70]

        if self.converged:
            results.extend([
                "✓ Method converged!",
                f"Root: {self.round_sig(self.root)}",
            ])
            try:
                g_at_root = self.evaluate_function(self.g, self.root)
                results.append(f"g(root) = {self.round_sig(g_at_root)}")
            except:
                results.append("g(root) = Could not evaluate")
        else:
            results.extend([
                "✗ Method did not converge",
                f"Approximate root: {self.round_sig(self.root)}"
            ])
            if self.error_message:
                results.append(f"Reason: {self.error_message}")

        results.append(f"Iterations: {self.iterations}")
        if self.relative_error is not None:
            results.append(f"Relative error: {self.round_sig(self.relative_error):.6f}%")
        results.append(f"Time: {self.execution_time:.6f}s")
        results.append("=" * 70)

        self.step_strings.append("\n".join(results))

        return self.get_results()

    def get_results(self):
        """Return results dictionary."""
        return {
            'root': self.round_sig(self.root) if self.root is not None else self.round_sig(self.x0),
            'iterations': self.iterations,
            'relative_error': self.round_sig(self.relative_error) if self.relative_error is not None else None,
            'execution_time': self.round_sig(self.execution_time),
            'converged': self.converged,
            'error_message': self.error_message,
            'iteration_history': self.iteration_history,
            'step_strings': self.step_strings,
            'significant_figures': self.last_significant_figures
        }

    def print_results(self):
        """Print formatted results."""
        for step in self.step_strings:
            print(step)