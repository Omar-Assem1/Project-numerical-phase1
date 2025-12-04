import { Routes } from '@angular/router';
import { LinearSystemSolverComponent } from './components/linear-system-solver/linear-system-solver';
import { RootFinderComponent } from './components/root-finder/root-finder';
import { HomeComponent } from './components/home/home';

export const routes: Routes = [
    { path: '', component: HomeComponent },
    { path: 'linear-system-solver', component: LinearSystemSolverComponent },
    { path: 'root-finder', component: RootFinderComponent }
];
