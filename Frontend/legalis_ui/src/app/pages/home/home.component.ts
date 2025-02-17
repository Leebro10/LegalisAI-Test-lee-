import { Component } from '@angular/core';
import { NavbarComponent } from '../navbar/navbar.component';
import { RouterLink } from '@angular/router';
import { TypebannerComponent } from '../typebanner/typebanner.component';
import { FeaturesComponent } from '../features/features.component';

@Component({
  selector: 'app-home',
  standalone:true,
  imports: [NavbarComponent,FeaturesComponent,TypebannerComponent,],  //Add imports here
  templateUrl: './home.component.html',
  styleUrl: './home.component.css'
})
export class HomeComponent {

}
