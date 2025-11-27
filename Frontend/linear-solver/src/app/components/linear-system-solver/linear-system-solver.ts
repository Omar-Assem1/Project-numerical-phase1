import { Component, OnInit, OnDestroy, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService, SolveRequest, SolveResponse } from '../../services/api.service';

@Component({
  selector: 'app-linear-system-solver',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './linear-system-solver.html',
  styleUrls: ['./linear-system-solver.css']
})
export class LinearSystemSolverComponent implements OnInit, OnDestroy {

  ///Start of Declaring variables and giving them some initial values

  numEquations: number = 3;
  matrix: string[][] = [];
  constants: string[] = [];
  method: string = 'gauss-elimination';
  luForm: string = 'doolittle';
  precision: number = 5;
  initialGuess: string[] = [];
  maxIterations: number = 50;
  tolerance: number = 0.00001;
  useScaling: boolean = false;
  stepByStep: boolean = true;

  // Results
  result: SolveResponse | null = null;
  steps: string[] = [];
  currentStep: number = 0;
  loading: boolean = false;
  errorMessage: string = '';

  // Simulation
  isPlaying: boolean = false;
  speedLevel: number = 5; // 1 (Slow) to 10 (Fast)
  private simulationInterval: any = null;

  methods = [
    { value: 'gauss-elimination', label: 'Gauss Elimination' },
    { value: 'gauss-jordan', label: 'Gauss-Jordan' },
    { value: 'lu-decomposition', label: 'LU Decomposition' },
    { value: 'jacobi', label: 'Jacobi Iteration' },
    { value: 'gauss-seidel', label: 'Gauss-Seidel' }
  ];

  /// End of declaring variables and giving them some initial values

  constructor(private apiService: ApiService, private cdr: ChangeDetectorRef) { }

  //On start initialize the matrix
  ngOnInit(): void {
    this.initializeMatrix();
  }

  ngOnDestroy(): void {
    this.stopSimulation();
  }

  //this is the method that initializes the matrix
  //it's very straightforward just filling the matrix in with zeros and empty cells for the view
  initializeMatrix(): void {
    this.matrix = Array(this.numEquations).fill(null).map(() => Array(this.numEquations).fill(''));
    this.constants = Array(this.numEquations).fill('');
    this.initialGuess = Array(this.numEquations).fill('0');
  }

  //this is the method responsible for adding equations
  // and filling up the new equations with empty spaces and zeros as in the initialization
  addEquation(): void {
      this.numEquations++;
      this.matrix.forEach(row => row.push(''));
      this.matrix.push(Array(this.numEquations).fill(''));
      this.constants.push('');
      this.initialGuess.push('0');
  }

  //this is the method responsible for removing an equation and removing its variables too
  removeEquation(): void {
    if (this.numEquations > 2) {
      this.numEquations--;
      this.matrix.forEach(row => row.pop());
      this.matrix.pop();
      this.constants.pop();
      this.initialGuess.pop();
    }
  }

  //this method is used to tell whether we are in an iterative method or not
  //used later on to turn on the iterative options
  isIterativeMethod(): boolean {
    return this.method === 'jacobi' || this.method === 'gauss-seidel';
  }

  //the method that gets called on pressing the solve button
  handleSolve(): void {
    this.loading = true;
    this.result = null;
    this.steps = [];
    this.currentStep = 0;
    this.stopSimulation();

    const payload: SolveRequest = {
      method: this.method,
      matrix: this.matrix,
      constants: this.constants,
      precision: this.precision,
      scaling: this.useScaling,
      stepByStep: this.stepByStep,
      luForm: this.luForm,
      initialGuess: this.initialGuess,
      maxIterations: this.maxIterations,
      tolerance: this.tolerance
    };

    this.apiService.solveSystem(payload).subscribe({
      next: (response) => {
        console.log('Component: Received response', response);
        this.result = response;
        if (response.steps && response.steps.length > 0) {
          this.steps = response.steps;
          this.currentStep = 0;
        }
        this.loading = false;
      },
      error: (error) => {
        this.errorMessage = error.error?.message || 'An error occurred';
        this.loading = false;
      }
    });
  }

  /// Start of the simulation handling methods

  previousStep(): void {
    if (this.currentStep > 0) {
      this.currentStep--;
    }
  }

  nextStep(): void {
    if (this.currentStep < this.steps.length - 1) {
      this.currentStep++;
    } else {
      this.stopSimulation();
    }
  }

  toggleSimulation(): void {
    if (this.isPlaying) {
      this.stopSimulation();
    } else {
      this.startSimulation();
    }
  }

  startSimulation(): void {
    if (this.currentStep >= this.steps.length - 1) {
      this.currentStep = 0;
    }

    this.isPlaying = true;
    const interval = 2200 - (this.speedLevel * 200); // Level 1 = 2000ms, Level 10 = 200ms
    this.simulationInterval = setInterval(() => {
      this.nextStep();
      this.cdr.detectChanges();
    }, interval);
  }

  stopSimulation(): void {
    this.isPlaying = false;
    if (this.simulationInterval) {
      clearInterval(this.simulationInterval);
      this.simulationInterval = null;
    }
  }

  onSpeedChange(): void {
    if (this.isPlaying) {
      this.stopSimulation();
      this.startSimulation();
    }
  }

  /// End of the simulation handling

  //a method that resets all equations
  clearAll(): void {
    this.initializeMatrix();
    this.result = null;
    this.steps = [];
    this.currentStep = 0;
    this.errorMessage = '';
    this.stopSimulation();
  }
}
