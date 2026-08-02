# CyberVision Face Recognition System

A modern, high-performance face recognition and biometric analysis system built with **Vite**, **Vanilla JS**, and **face-api.js**. The application features a futuristic cyber-themed UI, providing real-time face detection, facial expression analysis, age & gender estimation, and a built-in biometric enrollment system.

## 🚀 Features

- **Real-Time Detection**: Analyzes faces directly from the webcam stream.
- **Static Image Scanning**: Upload or drag-and-drop images for instant scanning and detection.
- **Biometric Enrollment**: Register faces into the local database and recognize them automatically in subsequent scans.
- **AI Core Modules**:
  - `TinyFaceDetector` & `SSDMobilenetv1` for fast and accurate face detection.
  - `FaceLandmark68Net` for drawing 68 facial landmark points.
  - `FaceRecognitionNet` for face matching and biometric registry.
  - `FaceExpressionNet` for real-time emotion detection (Happy, Sad, Angry, Surprised, etc.).
  - `AgeGenderNet` for estimating the person's age and biological gender.
- **Cyberpunk UI/UX**: An immersive interface featuring HUD overlays, scanlines, animations, and tabbed navigation.
- **Local Storage Integration**: Biometric profiles are saved securely in your browser's local storage.

## 🛠️ Technology Stack

- **Frontend**: HTML5, CSS3, Vanilla JavaScript (ES6+)
- **Build Tool**: [Vite](https://vitejs.dev/)
- **AI/ML Library**: [face-api.js](https://justadudewhohacks.github.io/face-api.js/docs/index.html) (running entirely client-side)

## 📦 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Kathirvel005/Face-Recognition_system.git
   cd Face-Recognition_system
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Run the development server**:
   ```bash
   npm run dev
   ```

4. Open your browser and navigate to the local URL provided by Vite (usually `http://localhost:5173`).

## 🎮 How to Use

1. **Live Camera Mode**: 
   - Grant camera permissions when prompted.
   - The AI will automatically detect your face, draw a bounding box, and overlay your emotion, age, and gender.
2. **Static Image Mode**:
   - Switch to the "Static Image" tab.
   - Drag & drop an image or click to upload one from your device.
   - Click "Initiate Scan" to run the detection model on the uploaded image.
3. **Biometric Enrollment**:
   - Enter a name in the "Subject Name" input field.
   - Click "Register Face" while your face is clearly visible (in webcam mode) or while a face is detected in the uploaded image.
   - The system will save your face encodings. The next time you are detected, the system will identify you by the registered name instead of "Unknown Subject".

## 📂 Project Structure

```
├── public/                 # Static assets and face-api.js model weights
│   └── models/             # Pre-trained AI models
├── src/                    # Source files
│   ├── main.js             # Core application logic, AI initialization, and UI handling
│   └── style.css           # Styling for the cyber-themed UI
├── index.html              # Main HTML structure
├── package.json            # Project metadata and dependencies
└── vite.config.js          # Vite configuration (base URL, build settings)
```

## ⚖️ License

This project is open-source and available under the MIT License.
