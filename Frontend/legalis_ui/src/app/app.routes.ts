import { Routes } from '@angular/router';
import { HomeComponent } from '../app/pages/home/home.component';
import { AboutComponent } from './pages/about/about.component';

export const routes: Routes = [
  { path: '', component: HomeComponent },
  {path: 'about', component: AboutComponent}, //Added enough routes
];
