from flask import Flask, request, jsonify
from flask_cors import CORS
import time

# Import your existing solver modules
from Itrativemethods.ItrativeMethods import ItrativeMethods
from nonlinear.ModifiedNewtonRaphsonMethod import ModifiedNewtonRaphsonMethod
from linear_system import LinearSystem
from nonlinear.falsePosition import falsePosition
from nonlinear.bisection import bisection
from nonlinear.fixedpoint import FixedPointMethod
from nonlinear.original_newton_raph import NewtonRaphsonMethod
from nonlinear.secant import Secant

from rank import SolutionType
from classes.forward_eliminator import ForwardEliminator
from classes.forward_eliminator_scaling import ForwardEliminatorScaling
from classes.system_solver import SystemSolver

from classes_for_gauss_jordan.gauss_jordan_eliminator import GaussJordanEliminator
from classes_for_gauss_jordan.gjscaling import GaussJordanEliminatorScaling
from classes_for_gauss_jordan.rref_solver import RREFSolver

from Dolittle.LUsolver import LUSolver
from chelosky_crout import Crout_LU, Chelosky_LU
from nonlinear import plotter

app = Flask(__name__)
CORS(app)


def convert_sympy(obj):
    from sympy import Basic

    if isinstance(obj, Basic):
        return str(obj)
    if isinstance(obj, list):
        return [convert_sympy(x) for x in obj]
    if isinstance(obj, dict):
        return {k: convert_sympy(v) for k, v in obj.items()}
    return obj


def format_error_message(error_msg):
    """Format error messages for better frontend display."""
    # Replace multiple newlines with single ones
    formatted = error_msg.replace('\n\n\n', '\n\n')
    
    # Split into lines for processing
    lines = formatted.split('\n')
    formatted_lines = []
    
    for line in lines:
        # Clean up extra spaces
        clean_line = ' '.join(line.split())
        if clean_line:  # Only add non-empty lines
            formatted_lines.append(clean_line)
    
    return '\n'.join(formatted_lines)


@app.route('/api/solve/linear', methods=['POST'])
def linear_solve():
    try:
        start_time = time.time()

        data = request.get_json()
        method = data.get('method')
        matrix_str = data.get('matrix', [])
        constants_str = data.get('constants', [])
        precision = data.get('precision', 5)
        scaling = data.get('scaling', False)
        step_by_step = data.get('stepByStep', True)
        lu_form = data.get('luForm', 'doolittle')
        initial_guess_str = data.get('initialGuess', [])
        max_iterations = data.get('maxIterations', 50)
        tolerance = float(data.get('tolerance', 0.00001))
        symbolic = data.get('symbolic', False)
        n = len(matrix_str)

        if not symbolic:
            matrix = [[float(val) if val.strip() else 0.0 for val in row] for row in matrix_str]
            constants = [float(val) if val.strip() else 0.0 for val in constants_str]
            augmented = [matrix[i][:] + [constants[i]] for i in range(n)]
        else:
            matrix = matrix_str
            constants = constants_str
            augmented = [matrix[i][:] + [constants[i]] for i in range(n)]

        if not symbolic:
            rankbro = SolutionType(augmented).gaussian_elimination()
            if (rankbro == 1):
                message = "INCONSISTENT"
            elif (rankbro == 2):
                message = "INFINITE NUMBER OF SOLUTIONS"
            else:
                message = "Unique Solution exists"
        else:
            message = "works"

        solution = None
        iterations = None
        steps = []

        if method == 'gauss-elimination' and not (
                message == "INCONSISTENT" or message == "INFINITE NUMBER OF SOLUTIONS"):
            if scaling:
                elim = ForwardEliminatorScaling(augmented.copy(), precision)
            else:
                elim = ForwardEliminator(augmented.copy(), precision)

            elim.eliminate()
            echelon, rank, pivots = elim.get_result()

            solver = SystemSolver(echelon, rank, n, pivots, precision)
            solution = solver.solve()

            steps = elim.step_strings + solver.step_strings

        elif method == 'gauss-jordan' and not (message == "INCONSISTENT" or message == "INFINITE NUMBER OF SOLUTIONS"):
            if scaling:
                elim = GaussJordanEliminatorScaling(augmented.copy(), precision)
            else:
                elim = GaussJordanEliminator(augmented.copy(), precision)

            elim.eliminate()
            rref, rank, pivots = elim.get_rref_result()

            solver = RREFSolver(rref, rank, n, pivots, precision)
            solution = solver.solve()
            steps = elim.step_strings + solver.step_strings

        elif method == 'lu-decomposition' and not (
                message == "INCONSISTENT" or message == "INFINITE NUMBER OF SOLUTIONS"):
            if lu_form == 'doolittle':
                solver = LUSolver(precision)
                solution = solver.solve(matrix.copy(), constants.copy())
                steps = solver.step_strings
            elif lu_form == 'crout':
                crout = Crout_LU(augmented.copy(), n, precision)
                solution = crout.solve()
                steps = crout.step_strings
            elif lu_form == 'cholesky':
                cholesky = Chelosky_LU(augmented.copy(), n, precision)
                solution = cholesky.solve()
                steps = cholesky.step_strings
            else:
                return jsonify({'error': 'Invalid LU form'}), 400

        elif method in ['jacobi', 'gauss-seidel']:
            initial_guess = [float(val) if val.strip() else 0.0 for val in initial_guess_str]
            solver = ItrativeMethods(n, matrix, constants, initial_guess, max_iterations, tolerance, precision)

            if method == 'jacobi':
                if not symbolic:
                    solver.print_iteration_formulas("jacobi")
                    solution = solver.jacobi()
                    steps = solver.getAnswer()
                else:
                    solution = solver.symbolic_iterations(max_iterations, method)
            else:
                if not symbolic:
                    solution = solver.seidel()
                    steps = solver.getAnswer()
                else:
                    solution = solver.symbolic_iterations(max_iterations, method)

        execution_time = (time.time() - start_time) * 1000

        if solution is not None and not symbolic:
            solution_str = [f"{val:.{precision}g}" for val in solution]
        elif solution is not None and symbolic:
            solution_str = convert_sympy(solution)
            steps = ''
        else:
            solution_str = None

        response = {
            'solution': solution_str,
            'executionTime': f"{execution_time:.10f}ms",
            'steps': steps if step_by_step else [],
            'message': message,
        }

        if iterations is not None:
            response['iterations'] = iterations
        if symbolic:
            response = convert_sympy(response)
        return jsonify(response), 200

    except ValueError as ve:
        return jsonify({'error': f'Invalid number format: {str(ve)}'}), 400
    except Exception as e:
        import traceback
        return jsonify({
            'error': 'Internal solver error',
            'details': str(e),
            'trace': traceback.format_exc()
        }), 500


@app.route('/api/plot', methods=['POST'])
def plot():
    try:
        data = request.get_json()
        method = data.get('method')
        function = data.get('equation')
        function = function.replace("^", "**")
        g_equation = data.get('gEquation', '')
        if g_equation:
            g_equation = g_equation.replace("^", "**")

        if method == 'fixed-point':
            return jsonify({'plotImage': plotter.get_plot_base64(g_equation.lower(), True)})
        else:
            return jsonify({'plotImage': plotter.get_plot_base64(function.lower())})
    except Exception as e:
        return jsonify({'error': f'Couldn\'t Plot: {str(e)}'}), 500


@app.route('/api/solve/nonlinear', methods=['POST'])
def nonlinear_solve():
    try:
        start_time = time.time()
        data = request.get_json()
        method = data.get('method')
        equation = data.get('equation')
        # Replace ^ with ** for Python evaluation
        equation = equation.replace("^", "**")

        xLower = float(data.get('xLower', 0))
        xUpper = float(data.get('xUpper', 0))
        x0 = float(data.get('x0', 0))
        x1 = float(data.get('x1', 0))
        precision = data.get('precision', 5)
        epsilon = float(data.get('eps', 0.00001))
        maxIterations = data.get('maxIterations', 50)
        step_by_step = data.get('stepByStep', True)

        # For fixed-point method
        g_equation = data.get('gEquation', '')
        if g_equation:
            g_equation = g_equation.replace("^", "**")

        solution = None
        steps = []
        approximateError = None
        iterations = 0
        significant_figures = None

        if method == 'bisection':
            bi = bisection(equation, xLower, xUpper, epsilon, maxIterations, precision)
            solution = bi.solve()
            steps = bi.step_strings
            approximateError = bi.approximateError
            iterations = bi.iterations

        elif method == 'false-position':
            fs = falsePosition(equation, xLower, xUpper, epsilon, maxIterations, precision)
            solution = fs.solve()
            steps = fs.step_strings
            approximateError = fs.approximateError
            iterations = fs.iterations

        elif method == 'fixed-point':
            fp = FixedPointMethod(
                equation_str=equation,
                initial_guess=x0,
                g_equation_str=g_equation if g_equation else None,
                epsilon=epsilon,
                max_iterations=maxIterations,
                precision=precision
            )
            result = fp.solve(show_steps=False)

            solution = result['root']
            steps = result['step_strings']
            approximateError = result['relative_error']
            iterations = result['iterations']
            significant_figures = result['significant_figures']

        elif method == 'newton':
            nr = NewtonRaphsonMethod(
                equation_str=equation,
                initial_guess=x0,
                epsilon=epsilon,
                max_iterations=maxIterations,
                precision=precision
            )
            result = nr.solve(show_steps=False)

            solution = result['root']
            steps = result['step_strings']
            approximateError = result['relative_error']
            iterations = result['iterations']
            significant_figures = result['significant_figures']

        elif method == 'modified-newton':
            multiplicity = data.get('multiplicity', None)
            if multiplicity is not None:
                multiplicity = int(multiplicity)

            mnr = ModifiedNewtonRaphsonMethod(
                equation_str=equation,
                initial_guess=x0,
                multiplicity=multiplicity,
                epsilon=epsilon,
                max_iterations=maxIterations,
                precision=precision
            )
            result = mnr.solve(show_steps=False)

            solution = result['root']
            steps = result['step_strings']
            approximateError = result['relative_error']
            iterations = result['iterations']
            significant_figures = result['significant_figures']

        elif method == 'secant':
            sec = Secant(
                f=equation,
                x0=x0,
                x1=x1,
                tol=epsilon,
                maxiter=maxIterations,
                precision=precision
            )
            solution = sec.solve()
            steps = sec.step_strings
            approximateError = sec.approximateError
            iterations = sec.iterations

        else:
            return jsonify({'error': f'Method {method} not supported'}), 400

        execution_time = (time.time() - start_time) * 1000

        response = {
            'root': solution,
            'iterations': iterations,
            'approximateError': approximateError,
            'executionTime': f"{execution_time:.10f}ms",
            'steps': steps if step_by_step else [],
        }

        # Add user-friendly message based on method results
        if method in ['fixed-point', 'newton', 'modified-newton']:
            result_obj = locals().get('result', {})
            if result_obj.get('converged'):
                response['message'] = '✓ Method converged successfully!'
            elif result_obj.get('error_message'):
                response['message'] = f"✗ {result_obj['error_message']}"
            else:
                response['message'] = '✗ Method did not converge'
        elif method in ['bisection', 'false-position']:
            solver_obj = locals().get('bi') or locals().get('fs')
            if solver_obj and hasattr(solver_obj, 'converged'):
                if solver_obj.converged:
                    response['message'] = '✓ Method converged successfully!'
                else:
                    response['message'] = '✗ Method did not converge'
        elif method == 'secant':
            sec_obj = locals().get('sec')
            if sec_obj and hasattr(sec_obj, 'converged'):
                if sec_obj.converged:
                    response['message'] = '✓ Method converged successfully!'
                else:
                    response['message'] = '✗ Method did not converge'

        if significant_figures is not None:
            if significant_figures == float('inf'):
                response['significantFigures'] = None
            elif significant_figures != significant_figures:  # Check for NaN
                response['significantFigures'] = None
            else:
                response['significantFigures'] = significant_figures

        return jsonify(response), 200

    except ValueError as ve:
        error_msg = str(ve)
        
        # Handle g(x) validation error specifically
        if "g(x) is not derived from f(x)" in error_msg:
            return jsonify({'error': 'g(x) is not derived from f(x)'}), 400
        elif "Invalid g(x) function" in error_msg or "🚫 Invalid g(x) Function" in error_msg:
            formatted_error = format_error_message(error_msg)
            return jsonify({
                'error': 'Invalid g(x) Function',
                'message': formatted_error,
                'type': 'validation_error'
            }), 400
        else:
            return jsonify({'error': f'Invalid input: {format_error_message(error_msg)}'}), 400
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'details': str(e),
            'trace': traceback.format_exc()
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)