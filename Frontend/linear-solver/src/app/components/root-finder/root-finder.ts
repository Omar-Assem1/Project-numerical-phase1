import { Component, OnInit, OnDestroy, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService, RootFindingRequest, RootFindingResponse, PlotRequest } from '../../services/api.service';

@Component({
  selector: 'app-root-finder',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './root-finder.html',
  styleUrls: ['./root-finder.css']
})
export class RootFinderComponent implements OnInit, OnDestroy {

  // Input parameters
  equation: string = '';
  method: string = 'bisection';
  xLower: string = '';
  xUpper: string = '';
  x0: string = '';
  x1: string = '';
  gEquation: string = '';
  precision: number = 5;
  eps: number = 0.00001;
  maxIterations: number = 50;
  stepByStep: boolean = true;

  // Results
  result: RootFindingResponse | null = null;
  steps: string[] = [];
  currentStep: number = 0;
  loading: boolean = false;
    errorMessage: string = '';

  // Plot
  plotImage: string | null = null;
  loadingPlot: boolean = false;
  plotErrorMessage: string = '';

  // Simulation (copied from Phase 1)
  isPlaying: boolean = false;
  speedLevel: number = 5;
  private simulationInterval: any = null;

  methods = [
    { value: 'bisection', label: 'Bisection' },
    { value: 'false-position', label: 'False-Position' },
    { value: 'fixed-point', label: 'Fixed Point' },
    { value: 'newton', label: 'Newton-Raphson' },
    { value: 'modified-newton', label: 'Modified Newton-Raphson' },
    { value: 'secant', label: 'Secant Method' }
  ];

  constructor(private apiService: ApiService, private cdr: ChangeDetectorRef) { }

  ngOnInit(): void {
    // Initialize with example equation
    this.equation = 'x^2 - 4';
    this.xLower = '-5';
    this.xUpper = '5';
    this.x0 = '1';
    this.x1 = '2';
  }

  ngOnDestroy(): void {
    this.stopSimulation();
  }

  // Check if method requires specific parameters
  requiresInterval(): boolean {
    return this.method === 'bisection' || this.method === 'false-position';
  }

  requiresSingleGuess(): boolean {
    return this.method === 'fixed-point' || this.method === 'newton' || this.method === 'modified-newton';
  }

  requiresTwoGuesses(): boolean {
    return this.method === 'secant';
  }

  // Handle solve button click
  handleSolve(): void {
    this.loading = true;
    this.result = null;
    this.steps = [];
    this.currentStep = 0;
    this.errorMessage = '';
    this.stopSimulation();

    const payload: RootFindingRequest = {
      method: this.method,
      equation: this.equation,
      precision: this.precision,
      eps: this.eps,
      maxIterations: this.maxIterations,
      stepByStep: this.stepByStep
    };

    // Add method-specific parameters
    if (this.requiresInterval()) {
      payload.xLower = this.xLower;
      payload.xUpper = this.xUpper;
    } else if (this.requiresSingleGuess()) {
      payload.x0 = this.x0;
      if (this.method === 'fixed-point' && this.gEquation) {  // ADD THIS
        payload.gEquation = this.gEquation;                    // ADD THIS
      }
    } else if (this.requiresTwoGuesses()) {
      payload.x0 = this.x0;
      payload.x1 = this.x1;
    }

    this.apiService.solveRootFinding(payload).subscribe({
      next: (response) => {
        console.log('Component: Received response', response);
        this.result = response;
        if (response.steps && response.steps.length > 0) {
          this.steps = response.steps;
          this.currentStep = 0;
        }
        this.loading = false;
        this.cdr.detectChanges();
      },
      error: (error) => {
        this.errorMessage = error.error?.error || error.error?.message || 'An error occurred while solving';
        this.loading = false;
        this.cdr.detectChanges();
      }
    });
  }

  // Handle plot button click
  handlePlot(): void {
    this.loadingPlot = true;
    this.plotImage = null;
    this.plotErrorMessage = '';

    const payload: PlotRequest = {
      method: this.method,
      equation: this.equation
    };

    this.apiService.getPlot(payload).subscribe({
      next: (response) => {
        console.log('Component: Received plot response', response);
        this.plotImage = response.plotImage;
        this.loadingPlot = false;
        this.cdr.detectChanges();
      },
      error: (error) => {
        this.plotErrorMessage = error.error?.message || 'An error occurred while generating plot';
        this.loadingPlot = false;
        this.cdr.detectChanges();
      }
    });
  }

  // Simulation control methods (copied from Phase 1)
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

  // Reset all inputs
  clearAll(): void {
    this.equation = '';
    this.xLower = '';
    this.xUpper = '';
    this.x0 = '';
    this.x1 = '';
    this.result = null;
    this.steps = [];
    this.currentStep = 0;
    this.errorMessage = '';
    this.plotImage = null;
    this.plotErrorMessage = '';
    this.stopSimulation();
  }
}
