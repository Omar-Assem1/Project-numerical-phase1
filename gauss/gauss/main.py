# main.py
from gauss.Itrativemethods.ItrativeMethods import IterativeMethods
from linear_system import LinearSystem
from classes.forward_eliminator import ForwardEliminator
from classes.forward_eliminator_scaling import ForwardEliminatorScaling
from classes.system_solver import SystemSolver

# Gauss-Jordan (assuming you have these with steps recording too)
from classes_for_gauss_jordan.gauss_jordan_eliminator import GaussJordanEliminator
from classes_for_gauss_jordan.gjscaling import GaussJordanEliminatorScaling
from classes_for_gauss_jordan.rref_solver import RREFSolver

# LU methods (you can later update them to collect steps too)
from Dolittle.LUsolver import LUSolver
from chelosky_crout import Crout_LU, Chelosky_LU


def main():
    # 1. Input n and precision
    n = int(input("Enter number of equations (n): "))
    precision = int(input("Enter desired precision (number of significant figures): "))
    X0 = [0,0,0,0]
    # Create system and input equations
    system = LinearSystem(n, precision)

    print(f"\nPlease enter {n} equations (each with {n} coefficients + 1 constant term):")
    print("Example: 2 1 3 10   → means 2x₁ + x₂ + 3x₃ = 10\n")

    while not system.is_complete():
        eq = input(f"Equation {system.current_row + 1}/{n}: ").strip()
        system.add_equation(eq)

    system.show("Your System (Augmented Matrix)")

    # Extract data
    augmented = system.copy_matrix()  # [A|b]
    A = [row[:n] for row in augmented]
    b = [row[n] for row in augmented]

    # Choose method
    print("\n" + "=" * 70)
    print("                LINEAR SYSTEM SOLVER")
    print("=" * 70)
    print("1. Gauss Elimination (Partial Pivoting)")
    print("2. Gauss-Jordan Elimination (RREF)")
    print("3. LU Doolittle + Partial Pivoting")
    print("4. Crout LU Decomposition")
    print("5. Cholesky Decomposition")
    print("6. Jacobi Method")
    print("6. Gauss-Seidel Method")

    print("-" * 70)
    choice = int(input("Enter choice (1-7): "))

    show_steps = input("\nShow step-by-step operations? (y/n): ").strip().lower() == 'y'

    # Master list to collect all steps
    steps = []

    print("\n" + "=" * 80)

    # ==================================================================
    if choice == 1:
        print("Gauss Elimination → (1) Partial Pivoting | (2) + Scaling")
        scaling_choice = int(input("Choose (1 or 2): "))

        if scaling_choice == 1:
            elim = ForwardEliminator(augmented.copy(), precision)
            elim.eliminate(steps)
        elif scaling_choice == 2:
            elim = ForwardEliminatorScaling(augmented.copy(), precision)
            elim.eliminate(steps)
        else:
            print("Invalid choice")
            return

        echelon, rank, pivots = elim.get_result()
        solver = SystemSolver(echelon, rank, n, pivots, precision)
        solution = solver.solve(steps)

    # ==================================================================
    elif choice == 2:
        print("Gauss-Jordan → (1) Partial Pivoting | (2) + Scaling")
        scaling_choice = int(input("Choose (1 or 2): "))

        if scaling_choice == 1:
            elim = GaussJordanEliminator(augmented.copy(), precision)
            elim.eliminate(steps)
        elif scaling_choice == 2:
            elim = GaussJordanEliminatorScaling(augmented.copy(), precision)
            elim.eliminate(steps)
        else:
            print("Invalid choice")
            return

        rref, rank, pivots = elim.get_rref_result()
        solver = RREFSolver(rref, rank, n, pivots, precision)
        solution = solver.solve(steps)

    # ==================================================================
    elif choice == 3:
        print("LU DECOMPOSITION (Doolittle + Partial Pivoting)")
        print("=" * 80)
        try:
            solver = LUSolver(precision)  # use user-defined precision
            solution = solver.solve(A.copy(), b.copy(), steps)  # pass steps list

            # Only print solution if not showing steps
            if not show_steps:
                print("UNIQUE SOLUTION:")
                print("-" * 50)
                for i, val in enumerate(solution, 1):
                    print(f"x{i} = {val:.{precision}g}")
                print("-" * 50)
        except Exception as e:
            steps.append(f"Error during LU decomposition: {e}")
            print("Error (possibly singular matrix):", e)

    # ==================================================================
    elif choice == 4:
        print("CROUT LU DECOMPOSITION (No Pivoting)")
        print("=" * 80)
        crout = Crout_LU(augmented.copy(), n, precision)
        solution = crout.solve(steps)

    elif choice == 5:
        print("CHOLESKY DECOMPOSITION (For SPD matrices only)")
        print("=" * 80)
        chelosky = Chelosky_LU(augmented.copy(), n, precision)
        solution = chelosky.solve(steps)
    elif choice == 6:
        tol = float(input("Enter tolerance: "))
        itr = int(input("Enter iterations: "))
        solver = IterativeMethods(n, A=A, b=b, X0=X0, max_iter=itr ,tol =tol, precision=precision)
        solver.jacobi(steps)
    elif choice == 7:
        tol = float(input("Enter tolerance: "))
        itr = int(input("Enter iterations: "))
        solver = IterativeMethods(n, A=A, b=b, X0=X0, max_iter=itr, tol=tol, precision=precision)
        solver.gauss_seidel(steps)
    else:
        print("Invalid choice!")
        return

    # ==================================================================
    # Final Output
    # ==================================================================
    if show_steps and steps:
        print("\n" + "="*80)
        print("               DETAILED STEP-BY-STEP SOLUTION")
        print("="*80)
        print("\n".join(steps))
        print("="*80)

    print("\nProgram finished. Goodbye!\n")


if __name__ == "__main__":
    main()