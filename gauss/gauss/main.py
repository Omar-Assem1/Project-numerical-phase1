from pyexpat.errors import messages

from flask import Flask, request, jsonify
from flask_cors import CORS
import time

# Import your existing solver modules
from .Itrativemethods.ItrativeMethods import ItrativeMethods
from .linear_system import LinearSystem
from .rank import SolutionType
from .classes.forward_eliminator import ForwardEliminator
from .classes.forward_eliminator_scaling import ForwardEliminatorScaling
from .classes.system_solver import SystemSolver

from .classes_for_gauss_jordan.gauss_jordan_eliminator import GaussJordanEliminator
from .classes_for_gauss_jordan.gjscaling import GaussJordanEliminatorScaling
from .classes_for_gauss_jordan.rref_solver import RREFSolver

from .Dolittle.LUsolver import LUSolver
from .chelosky_crout import Crout_LU, Chelosky_LU

app = Flask(__name__)
CORS(app)  # Enable CORS for Angular frontend


@app.route('/api/solve', methods=['POST'])
def solve_system():
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

        n = len(matrix_str)
        matrix = [[float(val) if val.strip() else 0.0 for val in row] for row in matrix_str]
        constants = [float(val) if val.strip() else 0.0 for val in constants_str]
        augmented = [matrix[i][:] + [constants[i]] for i in range(n)]


        rankbro = SolutionType(augmented).gaussian_elimination()
        if (rankbro == 1):
            message = "INCONSISTENT"
        elif (rankbro == 2):
            message = "INFINITE NUMBER OF SOLUTIONS"

        else:
            message = "Unique Solution exists"

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
                solver.print_iteration_formulas("jacobi")
                solution = solver.jacobi()

            else:
                solution = solver.seidel()

            steps = solver.getAnswer()  # ← Now this is correct!

        execution_time = (time.time() - start_time)*1000

        if solution is not None:
            solution_str = [f"{val:.{precision}g}" for val in solution]
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
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)