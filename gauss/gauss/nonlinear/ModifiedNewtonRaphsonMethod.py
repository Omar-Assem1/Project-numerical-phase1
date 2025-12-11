import math
import time
import sympy as sp
from decimal import Decimal, Context


class ModifiedNewtonRaphsonMethod:

    def __init__(self, equation_str, initial_guess, multiplicity=None,
                 epsilon=0.00001, max_iterations=50, significant_figures=None, precision=5):

        self.equation_str = equation_str
        self.x0 = initial_guess
        self.multiplicity = multiplicity  # momkn nzwdha ya mandobna
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.significant_figures = significant_figures if significant_figures is not None else precision
        self.precision = precision


        self.x = sp.Symbol('x')
        try:
            self.f = sp.sympify(equation_str)
            self.f_prime = sp.diff(self.f, self.x)
            self.f_double_prime = sp.diff(self.f_prime, self.x)
        except Exception as e:
            raise ValueError(f"Error parsing equation: {e}")


        self.root = None
        self.iterations = 0
        self.relative_error = None
        self.execution_time = 0
        self.iteration_history = []
        self.converged = False
        self.error_message = None
        self.step_strings = []  # For frontend display
        self.estimated_multiplicity = None  # Store if we estimate it

    def evaluate_function(self, func, x_val):
        try:
            result = float(func.subs(self.x, x_val))
            return result
        except Exception as e:
            raise ValueError(f"Error evaluating function at x={x_val}: {e}")

    def round_sig(self, x):
        if x == 0:
            return 0.0
        x_str = str(x)
        ctx = Context(prec=self.precision)
        return float(ctx.create_decimal(x_str).normalize())

    def calculate_relative_error(self, x_new, x_old):
        # Handle the case when x_new is zero
        if abs(x_new) < 1e-15:  
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

        rel_error = abs((x_new - x_old) / x_new)
        if rel_error == 0:
            return 15  # Maximum precision for float

        # Significant figures based on relative error
        if rel_error < 1e-10:
            return 10

        n = 2 - math.log10(2 * rel_error)
        return max(0, int(n))

    def estimate_multiplicity(self, x_val):
        """
        Estimate the multiplicity of a root using the formula:
        m ≈ f(x) * f'(x) / [f'(x)^2 - f(x) * f''(x)]
        
        Parameters:
        - x_val: Value at which to estimate multiplicity
        
        Returns:
        - Estimated multiplicity (rounded to nearest integer)
        """
        try:
            f_val = self.evaluate_function(self.f, x_val)
            f_prime_val = self.evaluate_function(self.f_prime, x_val)
            f_double_prime_val = self.evaluate_function(self.f_double_prime, x_val)

            # Check for division by zero
            denominator = f_prime_val**2 - f_val * f_double_prime_val
            if abs(denominator) < 1e-12:
                return None

            # Calculate estimated multiplicity
            m_estimate = (f_val * f_prime_val) / denominator
            
            # Round to nearest integer (multiplicity should be a whole number)
            m_rounded = round(abs(m_estimate))
            
            # Multiplicity should be at least 1
            return max(1, m_rounded)
        except Exception as e:
            return None

    def solve(self, show_steps=False):
        """
        1. If multiplicity m is known: x_{n+1} = x_n - m * f(x_n) / f'(x_n)
        2. If multiplicity unknown: x_{n+1} = x_n - f(x_n) / [f'(x_n) - f(x_n)*f''(x_n)/f'(x_n)]

        """
        start_time = time.time()
        self.step_strings = []  

        use_known_multiplicity = self.multiplicity is not None


        header = [
            "=" * 70,
            "Modified Newton-Raphson Method",
            "=" * 70,
            f"Equation: f(x) = {self.equation_str}",
            f"Derivative: f'(x) = {self.f_prime}",
        ]
        
        if use_known_multiplicity:
            header.append(f"Known Multiplicity: m = {self.multiplicity}")
            header.append(f"Formula: x_{{n+1}} = x_n - m * f(x_n) / f'(x_n)")
        else:
            header.append(f"Second Derivative: f''(x) = {self.f_double_prime}")
            header.append(f"Multiplicity: Unknown (will be estimated)")
            header.append(f"Formula: x_{{n+1}} = x_n - f(x_n) / [f'(x_n) - f(x_n)*f''(x_n)/f'(x_n)]")
        
        header.extend([
            f"Initial guess: x₀ = {self.x0}",
            f"Tolerance (ε): {self.epsilon}",
            f"Max iterations: {self.max_iterations}",
            "=" * 70
        ])
        self.step_strings.append("\n".join(header))

        x_old = self.x0
        self.iteration_history = []

        for i in range(self.max_iterations):
            try:
                # Evaluate function and derivatives at x_old
                f_val = self.evaluate_function(self.f, x_old)
                f_prime_val = self.evaluate_function(self.f_prime, x_old)
                
                 # Check if this is the root
                if abs(f_val) < 1e-12:
                    self.error_message = (
                        f"this is the root = {x_old:.{self.precision}f}. "
                        "f(x)=zero"
                    )
                    self.step_strings.append(self.error_message)
                    self.root = x_old
                    self.iterations = i
                    break

                # Check if derivative is zero
                if abs(f_prime_val) < 1e-12:
                    self.error_message = (
                        f"Derivative is zero at x = {x_old:.{self.precision}f}. "
                        "Cannot continue with Modified Newton-Raphson method."
                    )
                    self.step_strings.append(self.error_message)
                    self.root = x_old
                    self.iterations = i
                    break

                if use_known_multiplicity:
                    x_new = x_old - (self.multiplicity * f_val / f_prime_val)
                    method_used = f"Known multiplicity (m={self.multiplicity})"
                    f_double_prime_val = None  # Not needed for this approach
                else:
                    f_double_prime_val = self.evaluate_function(self.f_double_prime, x_old)
                    
                    denominator = f_prime_val - (f_val * f_double_prime_val / f_prime_val)
                    if abs(denominator) < 1e-12:
                        self.error_message = (
                            f"Denominator is zero at x = {x_old:.{self.precision}f}. "
                            "Cannot continue."
                        )
                        self.step_strings.append(self.error_message)
                        self.root = x_old
                        self.iterations = i
                        break
                    
                    x_new = x_old - (f_val / denominator)

                    m_estimate = self.estimate_multiplicity(x_old)
                    if m_estimate:
                        method_used = f"Estimated multiplicity (m≈{m_estimate})"
                        self.estimated_multiplicity = m_estimate
                    else:
                        method_used = "Unknown multiplicity formula"

                # Round x_new to specified precision
                x_new = self.round_sig(x_new)

                # Calculate errors
                rel_error = self.calculate_relative_error(x_new, x_old)
                f_new_val = self.evaluate_function(self.f, x_new)
                sig_figs = self.count_significant_figures(x_new, x_old)

                # Store iteration data
                iteration_data = {
                    'iteration': i + 1,
                    'x_old': x_old,
                    'f(x_old)': f_val,
                    'f_prime(x_old)': f_prime_val,
                    'x_new': x_new,
                    'f(x_new)': f_new_val,
                    'relative_error': rel_error,
                    'significant_figures': sig_figs,
                    'method_used': method_used
                }
                
                if not use_known_multiplicity:
                    iteration_data['f_double_prime(x_old)'] = f_double_prime_val
                
                self.iteration_history.append(iteration_data)

                # Build iteration step string
                iteration_step = [
                    f"Iteration {i + 1}:",
                    f"  x_{i} = {x_old:.{self.precision}f}",
                    f"  f(x_{i}) = {f_val:.{self.precision}e}",
                    f"  f'(x_{i}) = {f_prime_val:.{self.precision}e}",
                ]
                
                if not use_known_multiplicity:
                    iteration_step.append(f"  f''(x_{i}) = {f_double_prime_val:.{self.precision}e}")
                    iteration_step.append(f"  Method: {method_used}")
                
                iteration_step.extend([
                    f"  x_{i + 1} = {x_new:.{self.precision}f}",
                    f"  f(x_{i + 1}) = {f_new_val:.{self.precision}e}",
                    f"  |εₐ| = {rel_error:.6f}%",
                    f"  Significant figures: {sig_figs}"
                ])
                
                self.step_strings.append("\n".join(iteration_step))

                if show_steps:
                    print("\n".join(iteration_step))
                    print()

                # Check convergence based on relative error
                if rel_error < self.epsilon:
                    self.converged = True
                    self.root = x_new
                    self.iterations = i + 1
                    self.relative_error = rel_error
                    self.step_strings.append(f"✓ Converged! Relative error {rel_error:.6f}% < {self.epsilon}")
                    break

                # Check if significant figures requirement is met
                if self.significant_figures and sig_figs >= self.significant_figures:
                    self.converged = True
                    self.root = x_new
                    self.iterations = i + 1
                    self.relative_error = rel_error
                    self.step_strings.append(
                        f"✓ Converged! Significant figures {sig_figs} >= {self.significant_figures}")
                    break

                # Check if function value is close to zero
                if abs(f_new_val) < 1e-12:
                    self.converged = True
                    self.root = x_new
                    self.iterations = i + 1
                    self.relative_error = rel_error
                    self.step_strings.append(f"✓ Converged! f(x) ≈ 0")
                    break

                # Update x_old for next iteration
                x_old = x_new

            except Exception as e:
                self.error_message = f"Error during iteration {i + 1}: {e}"
                self.step_strings.append(self.error_message)
                break

        # Handle non-convergence
        if not self.converged and not self.error_message:
            self.root = x_new if 'x_new' in locals() else x_old
            self.iterations = self.max_iterations
            self.relative_error = rel_error if 'rel_error' in locals() else float('inf')
            self.error_message = (
                f"Method did not converge within {self.max_iterations} iterations. "
                f"Last approximation: {self.root:.10f}"
            )
            self.step_strings.append(self.error_message)

        self.execution_time = time.time() - start_time

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
                f"Approximate root: {self.root:.{self.precision}f}",
                f"f(root) = {self.evaluate_function(self.f, self.root):.{self.precision}e}"
            ])
        else:
            if (not (abs(f_val) < 1e-12)):
                results.append("✗ Method did not converge")
            if self.root:
                results.append(f"Last approximation: {self.root:.{self.precision}f}")

        results.append(f"Number of iterations: {self.iterations}")
        
        if self.estimated_multiplicity:
            results.append(f"Estimated multiplicity: {self.estimated_multiplicity}")
        
        if self.relative_error is not None:
            results.append(f"Approximate relative error: {self.relative_error:.6f}%")

        if self.converged and len(self.iteration_history) >= 1:
            sig_figs = self.iteration_history[-1]['significant_figures']
            results.append(f"Significant figures: {sig_figs}")

        results.append(f"Execution time: {self.execution_time:.6f} seconds")
        results.append("=" * 70)

        self.step_strings.append("\n".join(results))

        return self.get_results()

    def get_results(self):
        """
        Return results as a dictionary.
        
        Returns:
        - Dictionary containing all solution information
        """
        sig_figs = 0
        if self.converged and len(self.iteration_history) >= 1:
            last_iter = self.iteration_history[-1]
            sig_figs = last_iter['significant_figures']

        return {
            'root': self.root,
            'iterations': self.iterations,
            'relative_error': self.relative_error,
            'significant_figures': sig_figs,
            'execution_time': self.execution_time,
            'converged': self.converged,
            'error_message': self.error_message,
            'iteration_history': self.iteration_history,
            'step_strings': self.step_strings
        }

    def print_results(self):
        """Print formatted results to console."""
        for step in self.step_strings:
            print(step)


# Example usage
if __name__ == "__main__":
    print("Example 1: Known Multiplicity")
    print("-" * 70)
    solver1 = ModifiedNewtonRaphsonMethod(
        equation_str="(x-2)**3",
        initial_guess=3.0,
        multiplicity=3, 
        precision=6
    )
    result1 = solver1.solve(show_steps=True)
    print("\n")
    
    print("Example 2: Unknown Multiplicity (Estimated)")
    print("-" * 70)
    solver2 = ModifiedNewtonRaphsonMethod(
        equation_str="(x)**3-3*x+2",
        initial_guess=3.0,
        multiplicity=None,  
        precision=6
    )
    result2 = solver2.solve(show_steps=True)