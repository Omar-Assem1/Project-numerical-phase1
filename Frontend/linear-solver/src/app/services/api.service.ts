import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface SolveRequest {
  method: string;
  matrix: string[][];
  constants: string[];
  precision: number;
  scaling?: boolean;
  stepByStep?: boolean;
  luForm?: string;
  initialGuess?: string[];
  stoppingCondition?: string;
  maxIterations?: number;
  tolerance?: number;
  symbolic?: boolean;
}

export interface SolveResponse {
  solution?: string[];
  executionTime: string;
  iterations?: number;
  steps?: string[];
  message?: string;
  error?: string;
}

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private apiUrl = 'http://localhost:8080/api';

  constructor(private http: HttpClient) { }

  solveSystem(request: SolveRequest): Observable<SolveResponse> {

    //Ensuring the default values
    request.matrix = request.matrix.map(row =>
      row.map(cell => cell === '' ? '0' : cell)
    );
    request.constants = request.constants.map(cell =>
      cell === '' ? '0' : cell
    );
    request.initialGuess = request.initialGuess!.map(cell =>
      cell === '' ? '1' : cell
    );
    if (request.precision == null) {
      request.precision = 4;
    }
    //Setting a flag if the matrix is all symbols
    for (let i = 0; i < request.matrix.length; i++) {
      for (let j = 0; j < request.matrix[i].length; j++) {
        if (/^[a-zA-Z]$/.test(request.matrix[i][j])) {
          request.symbolic = true;
        }
        else{
          request.symbolic = false;
          break;
        }
      }
    }


    console.log('ApiService: Sending solve request to', `${this.apiUrl}/solve`, request);
    return this.http.post<SolveResponse>(`${this.apiUrl}/solve`, request);
  }
}
