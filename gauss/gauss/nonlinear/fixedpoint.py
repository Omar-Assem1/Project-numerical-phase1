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
                solutions = sp.solve(self.f, self.x)
                if not solutions:
                    self.g = self.x - self.f
                    self.g_equation_str = f"x - ({equation_str})"
                else:
                    self.g = solutions[0]
                    self.g_equation_str = str(self.g)
            else:
                # STRICT validation of the provided g(x)
                validation_result = self.validate_g_function_strict(equation_str, g_equation_str, initial_guess)

                if not validation_result['valid']:
                    # REJECT invalid rearrangements
                    error_msg = validation_result['message']
                    if validation_result.get('reason') == 'algebraic_mismatch':
                        raise ValueError(error_msg)
                    else:
                        raise ValueError(f"Invalid g(x) function: {error_msg}")
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

    def validate_g_function_strict(self, f_equation_str, g_equation_str, x0=None, tolerance=1e-6):
        """
        Strictly validate if g(x) is derived from f(x) = 0.

        For Fixed Point Method: f(x) = 0 must be equivalent to x = g(x)
        This means: f(x) = 0  ⟺  x - g(x) = 0
        Therefore: f(x) must equal x - g(x) (or a constant multiple)
        """
        result = {
            'valid': False,
            'message': '',
            'reason': '',
            'warning': None,
            'g_prime_at_x0': None,
            'converges': None
        }

        x = sp.Symbol('x')

        try:
            f = sp.sympify(f_equation_str)
            g = sp.sympify(g_equation_str)
        except Exception as e:
            result['message'] = f"Error parsing equations: {str(e)}"
            result['reason'] = 'parse_error'
            return result

        # Check 1: Algebraic equivalence f(x) = x - g(x)
        difference = sp.simplify(f - (x - g))

        if difference == 0:
            result['valid'] = True
            result['message'] = "✓ Valid: f(x) = x - g(x)"
        else:
            # Check if constant multiple: f(x) = k*(x - g(x))
            try:
                if (x - g) != 0:
                    ratio = sp.simplify(f / (x - g))
                    if ratio.is_number and ratio != 0:
                        result['valid'] = True
                        result['message'] = f"✓ Valid (scaled): f(x) = {ratio} * (x - g(x))"
                    else:
                        # Invalid - provide detailed error
                        result['valid'] = False
                        result['reason'] = 'algebraic_mismatch'
                        result['message'] = self._generate_error_message(f, g, x, f_equation_str)
                        return result
                else:
                    result['valid'] = False
                    result['reason'] = 'trivial_g'
                    result['message'] = "Invalid: g(x) = x (trivial, no iteration)"
                    return result
            except:
                # Numerical verification fallback
                if not self._numerical_verification(f, g, x, tolerance, x0):
                    result['valid'] = False
                    result['reason'] = 'algebraic_mismatch'
                    result['message'] = self._generate_error_message(f, g, x, f_equation_str)
                    return result
                else:
                    result['valid'] = True
                    result['message'] = "✓ Numerically verified"
                    result['warning'] = "Algebraic check inconclusive, passed numerical tests"

        # Check convergence condition
        if result['valid'] and x0 is not None:
            try:
                g_prime = sp.diff(g, x)
                g_prime_val = float(g_prime.subs(x, x0))
                result['g_prime_at_x0'] = g_prime_val

                if abs(g_prime_val) < 1:
                    result['converges'] = True
                    result['warning'] = (
                        f"✓ Convergence likely: |g'(x₀)| = {abs(g_prime_val):.6f} < 1"
                    )
                else:
                    result['converges'] = False
                    result['warning'] = (
                        f"⚠ May diverge: |g'(x₀)| = {abs(g_prime_val):.6f} ≥ 1"
                    )
            except Exception as e:
                result['warning'] = f"Could not check convergence: {str(e)}"

        return result

    def _numerical_verification(self, f, g, x, tolerance, x0):
        """Verify f(x) = x - g(x) numerically at multiple points."""
        test_points = [-10, -5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10]
        if x0 is not None:
            test_points.append(float(x0))

        valid_count = 0
        for test_x in test_points:
            try:
                f_val = float(f.subs(x, test_x))
                fixed_point_eq = test_x - float(g.subs(x, test_x))

                if abs(f_val - fixed_point_eq) < tolerance * (1 + abs(f_val)):
                    valid_count += 1
            except:
                continue

        return valid_count >= len(test_points) * 0.8  # 80% must pass

    def _generate_error_message(self, f, g, x, f_equation_str):
        """Generate simple error message with suggestions."""
        msg = "g(x) is not derived from f(x).\n\n"

        # Generate suggestions
        suggestions = self._generate_suggestions(f, x)
        if suggestions:
            msg += "Try these valid alternatives:\n"
            for i, sugg in enumerate(suggestions[:3], 1):
                msg += f"{i}. x = {sugg}\n"

        return msg

    def _generate_suggestions(self, f, x):
        """Generate valid rearrangement suggestions."""
        suggestions = []

        try:
            # Direct solve
            solutions = sp.solve(f, x)
            for sol in solutions[:2]:
                if sol.has(x) and sol != x:
                    suggestions.append(str(sol))

            # Newton-like: x = x - f(x)
            simple = sp.simplify(x - f)
            if simple != x and str(simple) not in suggestions:
                suggestions.append(str(simple))

            # Alternative: x = x + f(x)
            simple2 = sp.simplify(x + f)
            if simple2 != x and str(simple2) not in suggestions:
                suggestions.append(str(simple2))
        except:
            pass

        return suggestions

    def evaluate_function(self, func, x_val):
        """Safely evaluate a symbolic function at x_val."""
        try:
            return float(func.subs(self.x, x_val))
        except Exception as e:
            raise ValueError(f"Error evaluating function at x={x_val}: {e}")

    def round_sig(self, x):
        """Round number to specified precision."""
        if x == 0:
            return 0.0
        try:
            x_str = str(x)
            ctx = Context(prec=self.precision)
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
            f"Original equation: f(x) = {self.equation_str}",
            f"Rearranged form: x = g(x) = {self.g_equation_str}",
            f"Initial guess: x₀ = {self.x0}",
            f"Tolerance (ε): {self.epsilon}",
            f"Max iterations: {self.max_iterations}",
            "=" * 70
        ]
        self.step_strings.append("\n".join(header))

        # Validation warning
        if hasattr(self, 'validation_warning') and self.validation_warning:
            self.step_strings.append(self.validation_warning)

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

                try:
                    f_val = self.evaluate_function(self.f, x_new)
                    f_val = self.round_sig(f_val) if math.isfinite(f_val) else f_val
                except:
                    f_val = float('nan')

                sig_figs = self.count_significant_figures(x_new, x_old)

                x_old_rounded = self.round_sig(x_old)
                x_new_rounded = self.round_sig(x_new)

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
                    print("\n".join(iteration_step))

                # Convergence checks
                if math.isfinite(rel_error) and rel_error < self.epsilon:
                    self.converged = True
                    self.root = x_new_rounded
                    self.iterations = i + 1
                    self.relative_error = rel_error
                    self.step_strings.append(f"✓ Converged! Error {rel_error:.6f}% < {self.epsilon}")
                    break

                if self.significant_figures and sig_figs >= self.significant_figures:
                    self.converged = True
                    self.root = x_new_rounded
                    self.iterations = i + 1
                    self.relative_error = rel_error
                    self.step_strings.append(f"✓ Converged! {sig_figs} >= {self.significant_figures} sig figs")
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
                f"Root: {self.round_sig(self.root):.{self.precision}f}",
            ])
            try:
                f_at_root = self.evaluate_function(self.f, self.root)
                results.append(f"f(root) = {self.round_sig(f_at_root):.{self.precision}e}")
            except:
                results.append("f(root) = Could not evaluate")
        else:
            results.extend([
                "✗ Method did not converge",
                f"Best approximation: {self.round_sig(self.root):.{self.precision}f}"
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
        sig_figs = 0
        if self.converged and len(self.iteration_history) >= 1:
            sig_figs = self.iteration_history[-1]['significant_figures']

        return {
            'root': self.round_sig(self.root) if self.root is not None else self.round_sig(self.x0),
            'iterations': self.iterations,
            'relative_error': self.round_sig(self.relative_error) if self.relative_error is not None else None,
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