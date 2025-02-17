import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { NavbarComponent } from './pages/navbar/navbar.component';
import { FooterComponent } from './pages/footer/footer.component';
import { FeaturesComponent } from './pages/features/features.component';

@Component({
  selector: 'app-root',
  standalone:true,
  imports: [RouterOutlet,NavbarComponent,FooterComponent,FeaturesComponent,],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  title = 'legalis_ui';
}
