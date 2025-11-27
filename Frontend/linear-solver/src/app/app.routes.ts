import { Routes } from '@angular/router';
import {LinearSystemSolverComponent} from './components/linear-system-solver/linear-system-solver';

export const routes: Routes = [
    { path: '', redirectTo: 'linear-system-solver', pathMatch: 'full' },
    { path: 'linear-system-solver', component: LinearSystemSolverComponent }
];
