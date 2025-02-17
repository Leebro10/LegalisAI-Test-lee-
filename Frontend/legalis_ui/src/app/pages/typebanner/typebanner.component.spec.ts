import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TypebannerComponent } from './typebanner.component';

describe('TypebannerComponent', () => {
  let component: TypebannerComponent;
  let fixture: ComponentFixture<TypebannerComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TypebannerComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(TypebannerComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
