from flask import Flask, request, jsonify
from flask_cors import CORS
import time

# Import your existing solver modules
from .Itrativemethods.ItrativeMethods import IterativeMethods
from .linear_system import LinearSystem

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
    """
    Main endpoint to solve linear systems
    Expected JSON payload:
    {
        "method": "gauss-elimination" | "gauss-jordan" | "lu-decomposition" | "jacobi" | "gauss-seidel",
        "matrix": [["1", "2"], ["3", "4"]],
        "constants": ["5", "6"],
        "precision": 5,
        "scaling": false,
        "stepByStep": true,
        "luForm": "doolittle" | "crout" | "cholesky",
        "initialGuess": ["0", "0"],
        "maxIterations": 50,
        "tolerance": 0.00001
    }
    """
    try:
        start_time = time.time()

        # Parse request data
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
        tolerance = data.get('tolerance', 0.00001)

        # Convert string matrix to float
        n = len(matrix_str)
        matrix = [[float(val) if val else 0.0 for val in row] for row in matrix_str]
        constants = [float(val) if val else 0.0 for val in constants_str]

        # Create augmented matrix [A|b]
        augmented = [matrix[i] + [constants[i]] for i in range(n)]

        # Steps collection
        steps = []
        solution = None
        iterations = None

        # Route to appropriate solver
        if method == 'gauss-elimination':
            if scaling:
                elim = ForwardEliminatorScaling(augmented.copy(), precision)
            else:
                elim = ForwardEliminator(augmented.copy(), precision)

            elim.eliminate(steps)
            echelon, rank, pivots = elim.get_result()
            solver = SystemSolver(echelon, rank, n, pivots, precision)
            solution = solver.solve(steps)

        elif method == 'gauss-jordan':
            if scaling:
                elim = GaussJordanEliminatorScaling(augmented.copy(), precision)
            else:
                elim = GaussJordanEliminator(augmented.copy(), precision)

            elim.eliminate(steps)
            rref, rank, pivots = elim.get_rref_result()
            solver = RREFSolver(rref, rank, n, pivots, precision)
            solution = solver.solve(steps)

        elif method == 'lu-decomposition':
            if lu_form == 'doolittle':
                solver = LUSolver(precision)
                solution = solver.solve(matrix.copy(), constants.copy(), steps)
            elif lu_form == 'crout':
                crout = Crout_LU(augmented.copy(), n, precision)
                solution = crout.solve(steps)
            elif lu_form == 'cholesky':
                cholesky = Chelosky_LU(augmented.copy(), n, precision)
                solution = cholesky.solve(steps)
            else:
                return jsonify({'error': 'Invalid LU form'}), 400

        elif method == 'jacobi':
            initial_guess = [float(val) if val else 0.0 for val in initial_guess_str]
            solver = IterativeMethods(
                n=n,
                A=matrix,
                b=constants,
                X0=initial_guess,
                max_iter=max_iterations,
                tol=tolerance,
                precision=precision
            )
            solution, iterations = solver.jacobi(steps)

        elif method == 'gauss-seidel':
            initial_guess = [float(val) if val else 0.0 for val in initial_guess_str]
            solver = IterativeMethods(
                n=n,
                A=matrix,
                b=constants,
                X0=initial_guess,
                max_iter=max_iterations,
                tol=tolerance,
                precision=precision
            )
            solution, iterations = solver.gauss_seidel(steps)

        else:
            return jsonify({'error': 'Invalid method'}), 400

        # Calculate execution time
        execution_time = time.time() - start_time

        # Format solution
        if solution:
            solution_str = [f"{val:.{precision}g}" for val in solution]
        else:
            solution_str = None

        # Prepare response
        response = {
            'solution': solution_str,
            'executionTime': f"{execution_time:.9f}s",
            'steps': steps if step_by_step else []
        }

        if iterations is not None:
            response['iterations'] = iterations

        return jsonify(response), 200

    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Solver error: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'Linear System Solver API is running'}), 200


@app.route('/api/methods', methods=['GET'])
def get_methods():
    """Get available solving methods"""
    methods = [
        {
            'value': 'gauss-elimination',
            'label': 'Gauss Elimination',
            'supportsScaling': True,
            'isIterative': False
        },
        {
            'value': 'gauss-jordan',
            'label': 'Gauss-Jordan',
            'supportsScaling': True,
            'isIterative': False
        },
        {
            'value': 'lu-decomposition',
            'label': 'LU Decomposition',
            'supportsScaling': False,
            'isIterative': False,
            'forms': ['doolittle', 'crout', 'cholesky']
        },
        {
            'value': 'jacobi',
            'label': 'Jacobi Iteration',
            'supportsScaling': False,
            'isIterative': True
        },
        {
            'value': 'gauss-seidel',
            'label': 'Gauss-Seidel',
            'supportsScaling': False,
            'isIterative': True
        }
    ]
    return jsonify(methods), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)