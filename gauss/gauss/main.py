from pyexpat.errors import messages

from flask import Flask, request, jsonify
from flask_cors import CORS
import time
# Import your existing solver modules
from gauss.gauss.Itrativemethods.ItrativeMethods import ItrativeMethods
from gauss.gauss.linear_system import LinearSystem
from gauss.gauss.rank import SolutionType
from gauss.gauss.classes.forward_eliminator import ForwardEliminator
from gauss.gauss.classes.forward_eliminator_scaling import ForwardEliminatorScaling
from gauss.gauss.classes.system_solver import SystemSolver

from gauss.gauss.classes_for_gauss_jordan.gauss_jordan_eliminator import GaussJordanEliminator
from gauss.gauss.classes_for_gauss_jordan.gjscaling import GaussJordanEliminatorScaling
from gauss.gauss.classes_for_gauss_jordan.rref_solver import RREFSolver

from gauss.gauss.Dolittle.LUsolver import LUSolver
from gauss.gauss.chelosky_crout import Crout_LU, Chelosky_LU
from gauss.gauss.nonlinear import plotter
app = Flask(__name__)
CORS(app)  # Enable CORS for Angular frontend

def convert_sympy(obj):
    from sympy import Basic

    if isinstance(obj, Basic):     # Single SymPy expression
        return str(obj)
    if isinstance(obj, list):      # List of expressions
        return [convert_sympy(x) for x in obj]
    if isinstance(obj, dict):      # Dict with expressions
        return {k: convert_sympy(v) for k, v in obj.items()}
    return obj
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
        symbolic  = data.get('symbolic', False)
        n = len(matrix_str)
        if not symbolic:
            matrix = [[float(val) if val.strip() else 0.0 for val in row] for row in matrix_str]
            constants = [float(val) if val.strip() else 0.0 for val in constants_str]
            augmented = [matrix[i][:] + [constants[i]] for i in range(n)]
        else:
            matrix =matrix_str
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
        steps = []  # This will now hold full step strings

        if method == 'gauss-elimination' and not(message == "INCONSISTENT" or message == "INFINITE NUMBER OF SOLUTIONS"):
            if scaling:
                elim = ForwardEliminatorScaling(augmented.copy(), precision)
            else:
                elim = ForwardEliminator(augmented.copy(), precision)

            elim.eliminate()  # No more steps list
            echelon, rank, pivots = elim.get_result()

            solver = SystemSolver(echelon, rank, n, pivots, precision)
            solution = solver.solve()  # No steps param

            # Combine all step strings
            steps = elim.step_strings + solver.step_strings

        elif method == 'gauss-jordan' and not(message == "INCONSISTENT" or message == "INFINITE NUMBER OF SOLUTIONS"):
            if scaling:
                elim = GaussJordanEliminatorScaling(augmented.copy(), precision)
            else:
                elim = GaussJordanEliminator(augmented.copy(), precision)

            elim.eliminate()
            rref, rank, pivots = elim.get_rref_result()

            solver = RREFSolver(rref, rank, n, pivots, precision)
            solution = solver.solve()
            steps = elim.step_strings + solver.step_strings


        elif method == 'lu-decomposition'and not(message == "INCONSISTENT" or message == "INFINITE NUMBER OF SOLUTIONS"):

            if lu_form == 'doolittle':

                solver = LUSolver(precision)

                solution = solver.solve(matrix.copy(), constants.copy())

                steps = solver.step_strings  # ← Perfect list of full steps!
            elif lu_form == 'crout':
                crout = Crout_LU(augmented.copy(), n, precision)
                solution = crout.solve()
                steps = crout.step_strings  # assuming you update Crout_LU too
            elif lu_form == 'cholesky':
                cholesky = Chelosky_LU(augmented.copy(), n, precision)
                solution = cholesky.solve()
                steps = cholesky.step_strings
            else:
                return jsonify({'error': 'Invalid LU form'}), 400

        elif method in ['jacobi', 'gauss-seidel']:
            initial_guess = [float(val) if val.strip() else 0.0 for val in initial_guess_str]
            solver = ItrativeMethods(n, matrix, constants, initial_guess,max_iterations , tolerance, precision)

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
        execution_time = (time.time() - start_time)*1000

        if solution is not None and not symbolic:
            solution_str = [f"{val:.{precision}g}" for val in solution]
        elif solution is not None and symbolic:
            solution_str = convert_sympy(solution)
            steps = ''
        else:
            solution_str = None

        response = {
            'solution': solution_str,
            'executionTime': f"{execution_time:.10f}m",
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
    data = request.get_json()
    method = data.get('method')
    function = data.get('equation')
    function = function.replace("^", "**")
    try:

        if method == 'fixed-point':
            return jsonify({'plotImage':plotter.get_plot_base64(function,True)})
        else:
            return jsonify({'plotImage':plotter.get_plot_base64(function)})
    except Exception as e:
        return jsonify({'error': f'Couldn\'t Plot '}), 400



@app.route('/api/solve/nonlinear',methods=['POST'])
def nonlinear_solve():
    data = request.get_json()
    method = data.get('method')
    equation = data.get('equation')
    xLower = data.get('xLower',0)
    xUpper = data.get('xUpper',0)
    x0 = data.get('x0',0)
    x1 = data.get('x1',0)
    precision = data.get('precision',5)
    epsilon = data.get('eps', 0.00001)
    maxIterations = data.get('maxIterations',50)
    step_by_step = data.get('stepByStep')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
