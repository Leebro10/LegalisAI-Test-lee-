import { Component, AfterViewInit } from '@angular/core';

@Component({
  selector: 'app-typebanner',
  standalone: true,
  imports: [],
  templateUrl: './typebanner.component.html',
  styleUrls: ['./typebanner.component.css']
})
export class TypebannerComponent implements AfterViewInit {
  text: string = '';  // Text to display
  toRotate: string[] = [
    "Hi, I am LegalisAI!",
    "I am Your Legal Assistant",
    "I Am Proficient In MahRERA Laws",
    "How May I Assist You?"
  ];  // Array of texts to rotate
  currentTextIndex: number = 0;  // Tracks the current text
  isDeleting: boolean = false;  // Flag for delete mode
  typingSpeed: number = 150;  // Typing speed (milliseconds)
  deletingSpeed: number = 100;  // Deleting speed (milliseconds)
  pauseDuration: number = 3000;  // Duration to pause before deleting (milliseconds)
  typingInterval: any;  // Reference to the setInterval function
  typingCompleted: boolean = false;  // Flag to check if typing is complete

  constructor() {}

  ngAfterViewInit(): void {
    this.typeText();
  }

  // Sleep function for pauses
  sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // Main typing function
  async typeText() {
    let i = 0;
    let currentText = this.toRotate[this.currentTextIndex];
    let delta = this.typingSpeed;  // Typing speed for each character

    while (true) {
      if (this.isDeleting) {
        this.text = currentText.substring(0, i--);  // Deleting text
      } else {
        this.text = currentText.substring(0, i++);  // Typing text
      }

      // When full text is typed, pause before starting to delete
      if (!this.isDeleting && i === currentText.length+1) {
        this.typingCompleted = true;
        await this.sleep(this.pauseDuration);  // Pause before deleting
        this.isDeleting = true; // Begin deleting after the pause
      } else if (this.isDeleting && i === 0) {
        // Move to the next text after deleting
        this.isDeleting = false;
        this.typingCompleted = false;
        this.currentTextIndex = (this.currentTextIndex + 1) % this.toRotate.length; // Move to next text
        currentText = this.toRotate[this.currentTextIndex]; // Update the current text
        await this.sleep(500);  // Pause before typing the next string
        delta = this.typingSpeed;  // Restart typing at normal speed
      }

      // Adjust speed based on whether we are typing or deleting
      if (this.isDeleting && !this.typingCompleted) {
        delta = this.deletingSpeed;  // Speed for deleting
      }

      // Pause before typing the next character (typing or deleting)
      await this.sleep(delta); 
    }
  }

  ngOnDestroy(): void {
    // Cleanup the typing effect on component destruction
    clearInterval(this.typingInterval);
  }
}
