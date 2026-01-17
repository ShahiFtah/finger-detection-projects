# Finger Detection Projects

En samling prosjekter laget med **MediaPipe og OpenCV** – morsomme, Sci-Fi og nyttige applikasjoner med hånd- og fingerdeteksjon.

## Prosjekter inkludert

### 1. Pinch-to-Draw
- Tegn i luften med fingertupp
- Pinch tommel + pekefinger for å tegne
- Smooth linjer med neon-effekt

### 2. Minority Report Controller
- Håndbevegelser styrer mus og klikker
- Pinch → venstreklikk, lang pinch → høyreklikk, åpen hånd → scroll
- Neon fingertupper for feedback

### 3. Basic Finger detection
En enkel demonstrasjon av **fingerdeteksjon med MediaPipe og OpenCV**.  
Prosjektet viser hvordan systemet kan detektere fingertupper og tegne dots på hver fingertupp.

## Viktig fil

Dette prosjektet bruker modellen `hand_landmarker.task` fra MediaPipe for å detektere fingertuppene.  
Sørg for at filen ligger i samme mappe som `finger_detection.py` før du kjører scriptet.

## Krav
- Python 3.7+
- Kamera/webcam

## Installasjon

1. Sørg for at du har Python 3 installert
2. Sørg for godt lys for best deteksjon
3. Hold hånden i moderat avstand fra kameraet
4. Installer nødvendige pakker:
```bash
pip install opencv-python mediapipe numpy
