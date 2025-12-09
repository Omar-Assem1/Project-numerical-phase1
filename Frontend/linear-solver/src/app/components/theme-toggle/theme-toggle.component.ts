import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ThemeService } from '../../services/theme.service';

@Component({
    selector: 'app-theme-toggle',
    standalone: true,
    imports: [CommonModule],
    template: `
    <button
      (click)="toggleTheme()"
      class="fixed top-6 right-6 z-50 p-3 rounded-full bg-white dark:bg-gray-800 shadow-lg hover:shadow-xl transition-all duration-300 border border-gray-200 dark:border-gray-700 group"
      [attr.aria-label]="isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'">
      
      <!-- Sun Icon (Light Mode) -->
      <svg *ngIf="!isDarkMode" xmlns="http://www.w3.org/2000/svg" 
           class="h-6 w-6 text-yellow-500 group-hover:rotate-90 transition-transform duration-300" 
           fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
              d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
      </svg>
      
      <!-- Moon Icon (Dark Mode) -->
      <svg *ngIf="isDarkMode" xmlns="http://www.w3.org/2000/svg" 
           class="h-6 w-6 text-indigo-400 group-hover:rotate-12 transition-transform duration-300" 
           fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
              d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
      </svg>
    </button>
  `,
    styles: []
})
export class ThemeToggleComponent implements OnInit {
    isDarkMode = false;

    constructor(private themeService: ThemeService) { }

    ngOnInit(): void {
        this.themeService.darkMode$.subscribe(isDark => {
            this.isDarkMode = isDark;
        });
    }

    toggleTheme(): void {
        this.themeService.toggleDarkMode();
    }
}
