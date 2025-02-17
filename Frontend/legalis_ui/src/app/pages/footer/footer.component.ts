import { Component } from '@angular/core';

@Component({
  selector: 'app-footer',
  standalone:true,
  imports: [],
  templateUrl: './footer.component.html',
  styleUrl: './footer.component.css'
})
export class FooterComponent {
  currentTemps : any;
  ngOnInit():void{
    this.currentTemps = new Date().getFullYear();
  }
}
