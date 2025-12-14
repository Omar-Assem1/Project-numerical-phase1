import math
import time
import sympy as sp
from decimal import Decimal, Context


class FixedPointMethod:

    def __init__(self, equation_str, initial_guess, g_equation_str=None,
                 epsilon=0.00001, max_iterations=50, significant_figures=None, precision=5):

        self.equation_str = equation_str
        self.x0 = initial_guess
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.significant_figures = significant_figures if significant_figures is not None else precision
        self.precision = precision

        # Parse equations
        self.x = sp.Symbol('x')
        try:
            self.f = sp.sympify(equation_str)

            # If g(x) is not provided, try to solve for x automatically
            if g_equation_str is None or g_equation_str.strip() == '':
                # Try to solve f(x) = 0 for x to get x = g(x)
                solutions = sp.solve(self.f, self.x)
                if not solutions:
                    # Try a different approach: rearrange as x = x + f(x)
                    # or x = x - f(x) depending on the equation
                    self.g = self.x - self.f  # Simple rearrangement
                    self.g_equation_str = f"x - ({equation_str})"
                else:
                    # Use the first solution as g(x)
                    self.g = solutions[0]
                    self.g_equation_str = str(self.g)
            else:
                # Validate the provided g(x) before accepting it
                validation_result = self.validate_g_function(equation_str, g_equation_str, initial_guess)

                if not validation_result['valid']:
                    # Don't raise error, just warn
                    self.validation_warning = f"Warning: {validation_result['message']}"
                else:
                    self.validation_warning = validation_result.get('warning')

                self.g_equation_str = g_equation_str
                self.g = sp.sympify(g_equation_str)

            self.g_prime = sp.diff(self.g, self.x)
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
        self.step_strings = []

    def validate_g_function(self, f_equation_str, g_equation_str, x0=None):
        """
        Validate if g(x) is a correct rearrangement of f(x) = 0 for Fixed Point Iteration.

        Parameters:
        - f_equation_str: Original equation f(x)
        - g_equation_str: Rearranged form g(x)
        - x0: Initial guess (optional, for convergence check)

        Returns:
        - dict with keys: 'valid' (bool), 'message' (str), 'warning' (str or None),
          'g_prime_at_x0' (float or None), 'converges' (bool or None)
        """
        result = {
            'valid': False,
            'message': '',
            'warning': None,
            'g_prime_at_x0': None,
            'converges': None
        }

        x = sp.Symbol('x')

        try:
            # Parse the equations
            f = sp.sympify(f_equation_str)
            g = sp.sympify(g_equation_str)
        except Exception as e:
            result['message'] = f"Error parsing equations: {str(e)}"
            return result

        # Check 1: Verify that if x* is a root of f(x)=0, then x* = g(x*)
        try:
            # Get roots of f(x) = 0
            roots_f = sp.solve(f, x)

            if not roots_f:
                result['warning'] = (
                    "Could not find roots of f(x) = 0 to validate g(x). "
                    "The function will proceed but may not converge."
                )
                result['valid'] = True  # Allow it to proceed
                return result
            else:
                # Check if each root of f(x) is a fixed point of g(x)
                valid_root_found = False

                for root in roots_f:
                    try:
                        # Check if this is a real root
                        root_val = complex(root)
                        if abs(root_val.imag) > 1e-10:
                            continue  # Skip complex roots

                        root_float = float(root_val.real)

                        # Evaluate g(root)
                        g_at_root = float(g.subs(x, root))

                        # Check if root ≈ g(root)
                        if abs(root_float - g_at_root) < 0.001:
                            valid_root_found = True
                            break
                    except:
                        continue

                if not valid_root_found:
                    # Try numerical verification as backup
                    test_points = [-10, -5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10]
                    if x0 is not None:
                        test_points.append(float(x0))

                    for test_x in test_points:
                        try:
                            f_val = float(f.subs(x, test_x))
                            if abs(f_val) < 0.01:  # Near a root of f(x)
                                g_val = float(g.subs(x, test_x))
                                if abs(test_x - g_val) < 0.1:  # x ≈ g(x)
                                    valid_root_found = True
                                    break
                        except:
                            continue

                if not valid_root_found:
                    result['message'] = (
                        "The provided g(x) may not be a valid rearrangement. "
                        "For Fixed Point Method, if x* is a root of f(x) = 0, then x* must equal g(x*). "
                        "The method will attempt to solve but may not converge."
                    )
                    result['valid'] = True  # Allow it to proceed with warning
                    return result

        except Exception as e:
            # Don't fail on validation errors - just warn
            result['warning'] = f"Could not fully validate g(x): {str(e)}"

        # Check 2: Verify g(x) doesn't contain undefined operations
        try:
            test_val = 1.0 if x0 is None else float(x0)
            g_test = float(g.subs(x, test_val))

            if not sp.sympify(g).is_real:
                result['warning'] = "Warning: g(x) may produce complex values for some inputs."
        except Exception as e:
            result['message'] = (
                f"Error evaluating g(x) at test point: {str(e)}. "
                "The function may contain undefined operations."
            )
            result['valid'] = True  # Still allow it to try
            return result

        # Check 3: Check convergence condition |g'(x)| < 1 at x0
        if x0 is not None:
            try:
                g_prime = sp.diff(g, x)
                g_prime_val = float(g_prime.subs(x, x0))
                result['g_prime_at_x0'] = g_prime_val

                if abs(g_prime_val) < 1:
                    result['converges'] = True
                    result['warning'] = (
                        f"✓ Convergence condition satisfied: |g'(x₀)| = {abs(g_prime_val):.6f} < 1"
                    )
                else:
                    result['converges'] = False
                    result['warning'] = (
                        f"⚠ Convergence condition NOT satisfied: |g'(x₀)| = {abs(g_prime_val):.6f} ≥ 1. "
                        "The method may diverge or converge very slowly."
                    )
            except Exception as e:
                result['warning'] = f"Could not check convergence condition: {str(e)}"

        # If all checks pass
        result['valid'] = True
        result['message'] = "✓ Valid g(x) function! The rearrangement is mathematically correct."

        return result

    def evaluate_function(self, func, x_val):
        """Safely evaluate a symbolic function at x_val."""
        try:
            return float(func.subs(self.x, x_val))
        except Exception as e:
            raise ValueError(f"Error evaluating function at x={x_val}: {e}")

    def round_sig(self, x):
        """Round number to specified precision using significant figures."""
        if x == 0:
            return 0.0
        try:
            x_str = str(x)
            ctx = Context(prec=self.precision)  # Extra precision for rounding
            return float(ctx.create_decimal(x_str).normalize())
        except:
            return float(x)

    def calculate_relative_error(self, x_new, x_old):
        """Calculate approximate relative error."""
        if abs(x_new) < 1e-15:
            abs_error = abs(x_new - x_old)
            if abs_error < 1e-15:
                return 0.0
            else:
                return float('inf')

        return abs((x_new - x_old) / x_new) * 100

    def count_significant_figures(self, x_new, x_old):
        """Count the number of correct significant figures."""
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
        """Check if |g'(x)| < 1 at the initial guess."""
        try:
            g_prime_val = self.evaluate_function(self.g_prime, self.x0)
            return abs(g_prime_val) < 1, g_prime_val
        except:
            return False, None

    def solve(self, show_steps=False):
        """
        Solve the equation using Fixed Point Iteration.
        Always returns a root (best approximation) even if method doesn't converge.

        Parameters:
        - show_steps: If True, print each iteration step

        Returns:
        - Dictionary containing results
        """
        start_time = time.time()
        self.step_strings = []

        # Add initial information as a single header
        header = [
            "=" * 70,
            "Fixed Point Iteration Method",
            "=" * 70,
            f"Original equation: f(x) = {self.equation_str}",
            f"Rearranged form: x = g(x) = {self.g_equation_str}",
            f"Initial guess: x₀ = {self.x0}",
            f"Tolerance (ε): {self.epsilon}",
            f"Max iterations: {self.max_iterations}",
            "=" * 70
        ]
        self.step_strings.append("\n".join(header))

        # Add validation warning if exists
        if hasattr(self, 'validation_warning') and self.validation_warning:
            self.step_strings.append(self.validation_warning)

        # Check convergence condition
        converges, g_prime_val = self.check_convergence_condition()
        if g_prime_val is not None and not converges:
            warning = (
                f"Warning: |g'(x₀)| = {abs(g_prime_val):.6f} >= 1. "
                "The method may not converge."
            )
            self.step_strings.append(warning)

        x_old = self.x0
        x_new = self.x0  # Initialize to avoid undefined variable
        self.iteration_history = []
        last_valid_x = self.x0  # Store last valid value

        for i in range(self.max_iterations):
            try:
                # Calculate new x using g(x)
                x_new_raw = self.evaluate_function(self.g, x_old)

                # Check for NaN or infinity
                if not math.isfinite(x_new_raw):
                    self.error_message = f"Iteration {i + 1}: Function produced non-finite value (NaN or Inf)"
                    self.step_strings.append(self.error_message)
                    x_new = last_valid_x
                    break

                # Round x_new to specified precision
                x_new = self.round_sig(x_new_raw)
                last_valid_x = x_new  # Update last valid value

                # Calculate errors and round all values
                rel_error = self.calculate_relative_error(x_new, x_old)
                rel_error = self.round_sig(rel_error) if math.isfinite(rel_error) else rel_error

                try:
                    f_val = self.evaluate_function(self.f, x_new)
                    f_val = self.round_sig(f_val) if math.isfinite(f_val) else f_val
                except:
                    f_val = float('nan')

                sig_figs = self.count_significant_figures(x_new, x_old)

                # Round all values for storage
                x_old_rounded = self.round_sig(x_old)
                x_new_rounded = self.round_sig(x_new)

                # Store iteration data with rounded values
                iteration_data = {
                    'iteration': i + 1,
                    'x_old': x_old_rounded,
                    'x_new': x_new_rounded,
                    'f(x_new)': f_val,
                    'relative_error': rel_error,
                    'significant_figures': sig_figs
                }
                self.iteration_history.append(iteration_data)

                iteration_step = [
                    f"Iteration {i + 1}:",
                    f"  x_old = {x_old_rounded}",
                    f"  x_new = g(x_old) = {x_new_rounded}",
                    f"  f(x_new) = {f_val:.{self.precision}e}" if math.isfinite(f_val) else "  f(x_new) = undefined",
                    f"  |εₐ| = {rel_error:.6f}%" if math.isfinite(rel_error) else "  |εₐ| = N/A",
                    f"  Significant figures: {sig_figs}"
                ]
                self.step_strings.append("\n".join(iteration_step))

                if show_steps:
                    print(f"Iteration {i + 1}:")
                    print(f"  x_old = {x_old_rounded}")
                    print(f"  x_new = g(x_old) = {x_new_rounded}")
                    if math.isfinite(f_val):
                        print(f"  f(x_new) = {f_val:.{self.precision}e}")
                    else:
                        print(f"  f(x_new) = undefined")
                    if math.isfinite(rel_error):
                        print(f"  |εₐ| = {rel_error:.6f}%")
                    else:
                        print(f"  |εₐ| = N/A")
                    print(f"  Significant figures: {sig_figs}")
                    print()

                # Check convergence
                if math.isfinite(rel_error) and rel_error < self.epsilon:
                    self.converged = True
                    self.root = x_new_rounded
                    self.iterations = i + 1
                    self.relative_error = rel_error
                    self.step_strings.append(f"✓ Converged! Relative error {rel_error:.6f}% < {self.epsilon}")
                    break

                # Check if significant figures requirement is met
                if self.significant_figures and sig_figs >= self.significant_figures:
                    self.converged = True
                    self.root = x_new_rounded
                    self.iterations = i + 1
                    self.relative_error = rel_error
                    self.step_strings.append(
                        f"✓ Converged! Significant figures {sig_figs} >= {self.significant_figures}")
                    break

                # Check for oscillation or divergence
                if i > 5 and abs(x_new) > 1e10:
                    self.error_message = "Method appears to be diverging (values growing too large)"
                    self.step_strings.append(self.error_message)
                    self.root = self.round_sig(last_valid_x)
                    self.iterations = i + 1
                    self.relative_error = rel_error if math.isfinite(rel_error) else None
                    break

                x_old = x_new

            except Exception as e:
                self.error_message = f"Error during iteration {i + 1}: {str(e)}"
                self.step_strings.append(self.error_message)
                self.root = self.round_sig(last_valid_x)
                self.iterations = i + 1
                self.relative_error = None
                break

        # Ensure we always have a root value
        if self.root is None:
            self.root = self.round_sig(last_valid_x)

        if not self.converged and not self.error_message:
            self.iterations = self.max_iterations
            self.relative_error = self.round_sig(rel_error) if 'rel_error' in locals() and math.isfinite(
                rel_error) else None
            self.error_message = (
                f"Method did not converge within {self.max_iterations} iterations. "
                f"Returning best approximation."
            )
            self.step_strings.append(self.error_message)

        self.execution_time = self.round_sig(time.time() - start_time)

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
                f"Approximate root: {self.round_sig(self.root):.{self.precision}f}",
            ])
            try:
                f_at_root = self.evaluate_function(self.f, self.root)
                f_at_root_rounded = self.round_sig(f_at_root)
                results.append(f"f(root) = {f_at_root_rounded:.{self.precision}e}")
            except:
                results.append("f(root) = Could not evaluate")
        else:
            results.extend([
                "✗ Method did not converge",
                f"Best approximation: {self.round_sig(self.root):.{self.precision}f}"
            ])
            if self.error_message:
                results.append(f"Reason: {self.error_message}")

        results.append(f"Number of iterations: {self.iterations}")
        if self.relative_error is not None:
            results.append(f"Approximate relative error: {self.round_sig(self.relative_error):.6f}%")
        results.append(f"Execution time: {self.execution_time:.6f} seconds")
        results.append("=" * 70)

        self.step_strings.append("\n".join(results))

        return self.get_results()

    def get_results(self):
        """Return results as a dictionary. Always includes a root value."""
        sig_figs = 0
        if self.converged and len(self.iteration_history) >= 1:
            last_iter = self.iteration_history[-1]
            sig_figs = last_iter['significant_figures']

        # Round the root for final output
        rounded_root = self.round_sig(self.root) if self.root is not None else self.round_sig(self.x0)
        rounded_error = self.round_sig(self.relative_error) if self.relative_error is not None else None

        return {
            'root': rounded_root,
            'iterations': self.iterations,
            'relative_error': rounded_error,
            'significant_figures': sig_figs,
            'execution_time': self.round_sig(self.execution_time),
            'converged': self.converged,
            'error_message': self.error_message,
            'iteration_history': self.iteration_history,
            'step_strings': self.step_strings
        }

    def print_results(self):
        """Print formatted results."""
        for step in self.step_strings:
            print(step)