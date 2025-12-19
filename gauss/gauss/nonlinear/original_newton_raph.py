import math
import time
import sympy as sp
from decimal import Decimal, Context


class NewtonRaphsonMethod:


    def __init__(self, equation_str, initial_guess,
                 epsilon=0.00001, max_iterations=50, significant_figures=5):

        self.equation_str = equation_str
        self.x0 = initial_guess
        self.epsilon = epsilon * 100  # Convert decimal to percentage (0.00001 -> 0.001%)
        self.max_iterations = max_iterations
        self.significant_figures = significant_figures

        # Parse equation and compute derivative
        self.x = sp.Symbol('x')
        try:
            self.f = sp.sympify(equation_str)
            self.f_prime = sp.diff(self.f, self.x)
        except Exception as e:
            raise ValueError(f"Error parsing equation: {e}")

        # Results storage
        self.root = None
        self.iterations = 0
        self.relative_error = None
        self.execution_time = 0
        self.iteration_history = []
        self.converged = False
        self.error_message = None
        self.step_strings = []  # For frontend display

    def evaluate_function(self, func, x_val):
        """Safely evaluate a symbolic function at x_val."""
        try:
            result = float(func.subs(self.x, x_val))
            return result
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
        # Handle the case when x_new is zero
        if abs(x_new) < 1e-15:  # x_new is effectively zero
            # Use absolute error instead
            abs_error = abs(x_new - x_old)
            if abs_error < 1e-15:
                return 0.0  # Both are essentially zero, converged
            else:
                return float('inf')  # Can't calculate relative error

        return abs((x_new - x_old) / x_new) * 100

    def count_significant_figures(self, x_new, x_old):
        """Count the number of correct significant figures."""
        if x_new == 0:
            return 0

        rel_error_decimal = abs((x_new - x_old) / x_new)
        if rel_error_decimal == 0:
            return 15  # Maximum precision for float (was inf)

        # Significant figures based on relative error
        # |εa| = (0.5 × 10^(2-n)) %
        if rel_error_decimal < 1e-10:
            return 10

        # Use percentage value in the formula: rel_error_percentage = rel_error_decimal * 100
        rel_error_percentage = rel_error_decimal * 100
        n = 2 - math.log10(2 * rel_error_percentage)
        return max(0, int(n))

    def solve(self, show_steps=False):
        """
        Solve the equation using Newton-Raphson Method.

        Parameters:
        - show_steps: If True, print each iteration step

        Returns:
        - Dictionary containing results
        """
        start_time = time.time()
        self.step_strings = []  # Reset step strings

        # Add initial information as a single header
        header = [
            "=" * 70,
            "Newton-Raphson Method",
            "=" * 70,
            f"Equation: f(x) = {self.equation_str}",
            f"Derivative: f'(x) = {self.f_prime}",
            f"Initial guess: x₀ = {self.x0}",
            f"Tolerance (ε): {self.epsilon/100} ({self.epsilon}%)",
            f"Max iterations: {self.max_iterations}",
            "=" * 70
        ]
        self.step_strings.append("\n".join(header))

        x_old = self.x0
        self.iteration_history = []

        for i in range(self.max_iterations):
            try:
                # Evaluate function and derivative at x_old
                f_val = self.evaluate_function(self.f, x_old)
                f_prime_val = self.evaluate_function(self.f_prime, x_old)

                # Check if this is the root
                if abs(f_val) < 1e-12:
                    self.error_message = (
                        f"this is the root = {x_old:.{self.significant_figures}f}. "
                        "f(x)=zero"
                    )
                    self.step_strings.append(self.error_message)
                    self.root = x_old
                    self.iterations = i
                    break

                # Check if derivative is zero
                if abs(f_prime_val) < 1e-12:
                    self.error_message = (
                        f"Derivative is zero at x = {x_old:.{self.significant_figures}g}. "
                        "Cannot continue with Newton-Raphson method."
                    )
                    self.step_strings.append(self.error_message)
                    self.root = x_old
                    self.iterations = i
                    break

                # Newton-Raphson formula: x_new = x_old - f(x_old)/f'(x_old)
                x_new = x_old - (f_val / f_prime_val)

                # Round x_new to specified precision
                x_new = self.round_sig(x_new)

                # Calculate errors
                rel_error = self.calculate_relative_error(x_new, x_old)
                f_new_val = self.evaluate_function(self.f, x_new)

                # If function value is zero, set relative error to 0
                if abs(f_new_val) < 1e-12:
                    rel_error = 0

                # Store iteration data
                iteration_data = {
                    'iteration': i + 1,
                    'x_old': x_old,
                    'f(x_old)': f_val,
                    'f_prime(x_old)': f_prime_val,
                    'x_new': x_new,
                    'f(x_new)': f_new_val,
                    'relative_error': rel_error
                }
                self.iteration_history.append(iteration_data)

                # Add iteration as a single step string
                iteration_step = [
                    f"Iteration {i + 1}:",
                    f"  x_{i} = {x_old:.{self.significant_figures}g}",
                    f"  f(x_{i}) = {f_val:.{self.significant_figures}g}",
                    f"  f'(x_{i}) = {f_prime_val:.{self.significant_figures}g}",
                    f"  x_{i + 1} = x_{i} - f(x_{i})/f'(x_{i}) = {x_new:.{self.significant_figures}g}",
                    f"  f(x_{i + 1}) = {f_new_val:.{self.significant_figures}g}",
                    f"  |εₐ| = {rel_error:.6f}%"
                ]
                self.step_strings.append("\n".join(iteration_step))

                if show_steps:
                    print(f"Iteration {i + 1}:")
                    print(f"  x_{i} = {x_old:.10f}")
                    print(f"  f(x_{i}) = {f_val:.10e}")
                    print(f"  f'(x_{i}) = {f_prime_val:.10e}")
                    print(f"  x_{i + 1} = x_{i} - f(x_{i})/f'(x_{i}) = {x_new:.10f}")
                    print(f"  f(x_{i + 1}) = {f_new_val:.10e}")
                    print(f"  |εₐ| = {rel_error:.6f}%")
                    print(f"  Significant figures: {sig_figs}")
                    print()

                # Simple convergence check - stop when error <= epsilon OR max iterations
                if rel_error <= self.epsilon:
                    self.converged = True
                    self.root = x_new
                    self.iterations = i + 1
                    self.relative_error = rel_error
                    self.step_strings.append(f"✓ Converged! Error {rel_error:.6f}% <= {self.epsilon}%")
                    break

                # Check if function value is close to zero
                if abs(f_new_val) < 1e-12:
                    self.converged = True
                    self.root = x_new
                    self.iterations = i + 1
                    self.relative_error = 0  # Set to 0 when exact root found
                    self.step_strings.append(f"✓ Converged! f(x) ≈ 0")
                    break

                x_old = x_new

            except Exception as e:
                self.error_message = f"Error during iteration {i + 1}: {e}"
                self.step_strings.append(self.error_message)
                break

        if not self.converged and not self.error_message:
            self.root = x_new if 'x_new' in locals() else x_old
            self.iterations = self.max_iterations
            self.relative_error = rel_error if 'rel_error' in locals() else float('inf')
            self.error_message = (
                f"Method did not converge within {self.max_iterations} iterations. "
                f"Approximate root: {self.root:.{self.significant_figures}g}"
            )
            self.step_strings.append(self.error_message)

        self.execution_time = time.time() - start_time

        # Calculate significant figures at the END
        if self.relative_error == 0:
            final_sig_figs = self.significant_figures  # Use precision when exact solution found
        elif len(self.iteration_history) >= 2:
            # Calculate from last two iterations
            last_x = self.iteration_history[-1]['x_new']
            prev_x = self.iteration_history[-2]['x_new']
            final_sig_figs = self.count_significant_figures(last_x, prev_x)
        else:
            final_sig_figs = self.significant_figures

        # Add final results as a single step
        results = [
            "",
            "=" * 70,
            "RESULTS",
            "=" * 70
        ]

        if self.converged:
            results.extend([
                "✓ Method converged successfully!",
                f"Approximate root: {self.root:.{self.significant_figures}g}",
                f"f(root) = {self.evaluate_function(self.f, self.root):.{self.significant_figures}g}"
            ])
        else:
            results.append("✗ Method did not converge")
            if self.root:
                results.append(f"Approximate root: {self.root:.{self.significant_figures}g}")
            if self.error_message:
                results.append(f"Reason: {self.error_message}")

        results.append(f"Number of iterations: {self.iterations}")
        if self.relative_error is not None:
            results.append(f"Approximate relative error: {self.relative_error:.6f}%")

        results.append(f"Significant figures: {final_sig_figs}")
        results.append(f"Execution time: {self.execution_time:.6f} seconds")
        results.append("=" * 70)

        self.step_strings.append("\n".join(results))

        return self.get_results()

    def get_results(self):
        """Return results as a dictionary."""
        # Calculate final significant figures for return
        if self.relative_error == 0:
            final_sig_figs = self.significant_figures  # Use precision when exact solution found
        elif len(self.iteration_history) >= 2:
            # Calculate from last two iterations
            last_x = self.iteration_history[-1]['x_new']
            prev_x = self.iteration_history[-2]['x_new']
            final_sig_figs = self.count_significant_figures(last_x, prev_x)
        else:
            final_sig_figs = self.significant_figures
            
        return {
            'root': self.root,
            'iterations': self.iterations,
            'relative_error': round(self.relative_error, 6) if self.relative_error is not None else None,
            'execution_time': self.execution_time,
            'converged': self.converged,
            'error_message': self.error_message,
            'iteration_history': self.iteration_history,
            'step_strings': self.step_strings,
            'significant_figures': final_sig_figs
        }

    def print_results(self):
        """Print formatted results."""
        for step in self.step_strings:
            print(step)