### Trembling laser meter
This is a project from graduate course of Mechanical Engineering with ephasis in metrology and instrumantation from Federal Univeristy of Santa Catarina.

The project aims to detect a laser projected on a screen inside a webcam's field of view and detect problems of trembling hands.

## How to use

Initially you must identify the projector output port and define in main in ``display``. Afterward initialize the system, the algorithm will
identify the screen information and print four Aruco patterns to warp perspective of camera view.
    
- Keymap:
  - **Q**: quit program
  - **space**: warp perspective
  - **R**: reset perspective
  - **S**: save image
  - **Enter**: start trembling test