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
        console.log('ApiService: Sending solve request to', `${this.apiUrl}/solve`, request);
        return this.http.post<SolveResponse>(`${this.apiUrl}/solve`, request);
    }
}
