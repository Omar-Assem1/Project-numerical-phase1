"""
Examples using Fixed Point and Newton-Raphson methods
with exponential and trigonometric functions
"""

from fixedpoint import FixedPointMethod
from original_newton_raph import NewtonRaphsonMethod


def exponential_and_trig_examples():
    """
    Comprehensive examples with exponential (exp) and trigonometric (cos, sin) functions
    """

    print("=" * 80)
    print("EXPONENTIAL AND TRIGONOMETRIC FUNCTIONS - TEST EXAMPLES")
    print("=" * 80)

    # ========================================================================
    # EXPONENTIAL EXAMPLES
    # ========================================================================

    # Example 1: e^x - 3x = 0 using Newton-Raphson
    print("\n" + "=" * 80)
    print("EXAMPLE 1: e^x - 3x = 0 (Newton-Raphson)")
    print("=" * 80)

    solver1 = NewtonRaphsonMethod(
        equation_str="exp(x) - 3*x",
        initial_guess=1.5,
        epsilon=0.00001,
        max_iterations=50
    )
    solver1.solve(show_steps=True)
    solver1.print_results()

    # Example 2: e^x - 3x = 0 using Fixed Point (x = e^x / 3)
    print("\n" + "=" * 80)
    print("EXAMPLE 2: e^x - 3x = 0 (Fixed Point: x = e^x / 3)")
    print("=" * 80)

    solver2 = FixedPointMethod(
        equation_str="exp(x) - 3*x",
        g_equation_str="exp(x) / 3",
        initial_guess=1.5,
        epsilon=0.00001,
        max_iterations=50
    )
    solver2.solve(show_steps=True)
    solver2.print_results()

    # Example 3: e^(-x) - x = 0 using Newton-Raphson
    print("\n" + "=" * 80)
    print("EXAMPLE 3: e^(-x) - x = 0 (Newton-Raphson)")
    print("=" * 80)

    solver3 = NewtonRaphsonMethod(
        equation_str="exp(-x) - x",
        initial_guess=0.5,
        epsilon=0.00001,
        max_iterations=50
    )
    solver3.solve(show_steps=True)
    solver3.print_results()

    # Example 4: e^(-x) - x = 0 using Fixed Point (x = e^(-x))
    print("\n" + "=" * 80)
    print("EXAMPLE 4: e^(-x) - x = 0 (Fixed Point: x = e^(-x))")
    print("=" * 80)

    solver4 = FixedPointMethod(
        equation_str="exp(-x) - x",
        g_equation_str="exp(-x)",
        initial_guess=0.5,
        epsilon=0.00001,
        max_iterations=50
    )
    solver4.solve(show_steps=True)
    solver4.print_results()

    # Example 5: x*e^x - 1 = 0 using Newton-Raphson
    print("\n" + "=" * 80)
    print("EXAMPLE 5: x*e^x - 1 = 0 (Newton-Raphson)")
    print("=" * 80)

    solver5 = NewtonRaphsonMethod(
        equation_str="x*exp(x) - 1",
        initial_guess=0.5,
        epsilon=0.00001,
        max_iterations=50
    )
    solver5.solve(show_steps=True)
    solver5.print_results()

    # ========================================================================
    # TRIGONOMETRIC EXAMPLES (COSINE)
    # ========================================================================

    # Example 6: cos(x) - x = 0 using Newton-Raphson
    print("\n" + "=" * 80)
    print("EXAMPLE 6: cos(x) - x = 0 (Newton-Raphson)")
    print("=" * 80)

    solver6 = NewtonRaphsonMethod(
        equation_str="cos(x) - x",
        initial_guess=0.5,
        epsilon=0.00001,
        max_iterations=50
    )
    solver6.solve(show_steps=True)
    solver6.print_results()

    # Example 7: cos(x) - x = 0 using Fixed Point (x = cos(x))
    print("\n" + "=" * 80)
    print("EXAMPLE 7: cos(x) - x = 0 (Fixed Point: x = cos(x))")
    print("=" * 80)

    solver7 = FixedPointMethod(
        equation_str="cos(x) - x",
        g_equation_str="cos(x)",
        initial_guess=0.5,
        epsilon=0.00001,
        max_iterations=50
    )
    solver7.solve(show_steps=True)
    solver7.print_results()

    # Example 8: x*cos(x) - 1 = 0 using Newton-Raphson
    print("\n" + "=" * 80)
    print("EXAMPLE 8: x*cos(x) - 1 = 0 (Newton-Raphson)")
    print("=" * 80)

    solver8 = NewtonRaphsonMethod(
        equation_str="x*cos(x) - 1",
        initial_guess=1.0,
        epsilon=0.00001,
        max_iterations=50
    )
    solver8.solve(show_steps=True)
    solver8.print_results()

    # Example 9: cos(x) - 2*x = 0 using Fixed Point (x = cos(x)/2)
    print("\n" + "=" * 80)
    print("EXAMPLE 9: cos(x) - 2*x = 0 (Fixed Point: x = cos(x)/2)")
    print("=" * 80)

    solver9 = FixedPointMethod(
        equation_str="cos(x) - 2*x",
        g_equation_str="cos(x)/2",
        initial_guess=0.3,
        epsilon=0.00001,
        max_iterations=50
    )
    solver9.solve(show_steps=True)
    solver9.print_results()

    # ========================================================================
    # TRIGONOMETRIC EXAMPLES (SINE)
    # ========================================================================

    # Example 10: sin(x) - x/2 = 0 using Newton-Raphson
    print("\n" + "=" * 80)
    print("EXAMPLE 10: sin(x) - x/2 = 0 (Newton-Raphson)")
    print("=" * 80)

    solver10 = NewtonRaphsonMethod(
        equation_str="sin(x) - x/2",
        initial_guess=2.0,
        epsilon=0.00001,
        max_iterations=50
    )
    solver10.solve(show_steps=True)
    solver10.print_results()

    # Example 11: sin(x) - x/2 = 0 using Fixed Point (x = 2*sin(x))
    print("\n" + "=" * 80)
    print("EXAMPLE 11: sin(x) - x/2 = 0 (Fixed Point: x = 2*sin(x))")
    print("=" * 80)

    solver11 = FixedPointMethod(
        equation_str="sin(x) - x/2",
        g_equation_str="2*sin(x)",
        initial_guess=1.5,
        epsilon=0.00001,
        max_iterations=50
    )
    solver11.solve(show_steps=True)
    solver11.print_results()

    # Example 12: x*sin(x) - 1 = 0 using Newton-Raphson
    print("\n" + "=" * 80)
    print("EXAMPLE 12: x*sin(x) - 1 = 0 (Newton-Raphson)")
    print("=" * 80)

    solver12 = NewtonRaphsonMethod(
        equation_str="x*sin(x) - 1",
        initial_guess=1.5,
        epsilon=0.00001,
        max_iterations=50
    )
    solver12.solve(show_steps=True)
    solver12.print_results()

    # ========================================================================
    # COMBINED EXAMPLES (EXPONENTIAL + TRIGONOMETRIC)
    # ========================================================================

    # Example 13: e^x - cos(x) = 0 using Newton-Raphson
    print("\n" + "=" * 80)
    print("EXAMPLE 13: e^x - cos(x) = 0 (Newton-Raphson)")
    print("=" * 80)

    solver13 = NewtonRaphsonMethod(
        equation_str="exp(x) - cos(x)",
        initial_guess=0.0,
        epsilon=0.00001,
        max_iterations=50
    )
    solver13.solve(show_steps=True)
    solver13.print_results()

    # Example 14: e^x + sin(x) - 1 = 0 using Newton-Raphson
    print("\n" + "=" * 80)
    print("EXAMPLE 14: e^x + sin(x) - 1 = 0 (Newton-Raphson)")
    print("=" * 80)

    solver14 = NewtonRaphsonMethod(
        equation_str="exp(x) + sin(x) - 1",
        initial_guess=0.0,
        epsilon=0.00001,
        max_iterations=50
    )
    solver14.solve(show_steps=True)
    solver14.print_results()

    # Example 15: sin(x) - e^(-x) = 0 using Newton-Raphson
    print("\n" + "=" * 80)
    print("EXAMPLE 15: sin(x) - e^(-x) = 0 (Newton-Raphson)")
    print("=" * 80)

    solver15 = NewtonRaphsonMethod(
        equation_str="sin(x) - exp(-x)",
        initial_guess=0.5,
        epsilon=0.00001,
        max_iterations=50
    )
    solver15.solve(show_steps=True)
    solver15.print_results()

    # Example 16: cos(x) - x*e^(-x) = 0 using Newton-Raphson
    print("\n" + "=" * 80)
    print("EXAMPLE 16: cos(x) - x*e^(-x) = 0 (Newton-Raphson)")
    print("=" * 80)

    solver16 = NewtonRaphsonMethod(
        equation_str="cos(x) - x*exp(-x)",
        initial_guess=1.0,
        epsilon=0.00001,
        max_iterations=50
    )
    solver16.solve(show_steps=True)
    solver16.print_results()

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print("\n" + "=" * 80)
    print("SUMMARY OF ALL EXAMPLES")
    print("=" * 80)
    print("\nExponential Functions:")
    print("  1. e^x - 3x = 0 (Newton-Raphson)")
    print("  2. e^x - 3x = 0 (Fixed Point)")
    print("  3. e^(-x) - x = 0 (Newton-Raphson)")
    print("  4. e^(-x) - x = 0 (Fixed Point)")
    print("  5. x*e^x - 1 = 0 (Newton-Raphson)")

    print("\nTrigonometric Functions (Cosine):")
    print("  6. cos(x) - x = 0 (Newton-Raphson)")
    print("  7. cos(x) - x = 0 (Fixed Point)")
    print("  8. x*cos(x) - 1 = 0 (Newton-Raphson)")
    print("  9. cos(x) - 2*x = 0 (Fixed Point)")

    print("\nTrigonometric Functions (Sine):")
    print(" 10. sin(x) - x/2 = 0 (Newton-Raphson)")
    print(" 11. sin(x) - x/2 = 0 (Fixed Point)")
    print(" 12. x*sin(x) - 1 = 0 (Newton-Raphson)")

    print("\nCombined (Exponential + Trigonometric):")
    print(" 13. e^x - cos(x) = 0 (Newton-Raphson)")
    print(" 14. e^x + sin(x) - 1 = 0 (Newton-Raphson)")
    print(" 15. sin(x) - e^(-x) = 0 (Newton-Raphson)")
    print(" 16. cos(x) - x*e^(-x) = 0 (Newton-Raphson)")

    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    exponential_and_trig_examples()