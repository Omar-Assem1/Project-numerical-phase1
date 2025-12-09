import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';

@Component({
    selector: 'app-home',
    standalone: true,
    imports: [CommonModule],
    templateUrl: './home.html',
    styleUrls: ['./home.css']
})
export class HomeComponent {
    constructor(private router: Router) { }

    navigateTo(route: string): void {
        this.router.navigate([route]);
    }
}
