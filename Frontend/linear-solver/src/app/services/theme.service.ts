import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

@Injectable({
    providedIn: 'root'
})
export class ThemeService {
    private darkModeSubject = new BehaviorSubject<boolean>(this.getInitialTheme());
    public darkMode$ = this.darkModeSubject.asObservable();

    constructor() {
        this.applyTheme(this.darkModeSubject.value);
    }

    private getInitialTheme(): boolean {
        // Check localStorage first
        const savedTheme = localStorage.getItem('darkMode');
        if (savedTheme !== null) {
            return savedTheme === 'true';
        }
        // Fallback to system preference
        return window.matchMedia('(prefers-color-scheme: dark)').matches;
    }

    toggleDarkMode(): void {
        const newValue = !this.darkModeSubject.value;
        this.darkModeSubject.next(newValue);
        this.applyTheme(newValue);
        localStorage.setItem('darkMode', String(newValue));
    }

    private applyTheme(isDark: boolean): void {
        if (isDark) {
            document.documentElement.classList.add('dark');
        } else {
            document.documentElement.classList.remove('dark');
        }
    }

    get isDarkMode(): boolean {
        return this.darkModeSubject.value;
    }
}
