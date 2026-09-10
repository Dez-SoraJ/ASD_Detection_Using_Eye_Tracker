# ASD_Detection_Using_Eye_Tracker
The primary goal is to create a reliable predictive tool that can act as a screening mechanism, not a diagnostic one. It aims to identify individuals toddlers  who are at a high risk for ASD, flagging them for a comprehensive evaluation by a qualified healthcare professional. Early identification is critical as it enables access to early intervention therapies, which can significantly improve long-term outcomes. 

► Eye Tracking Dataset

o Contains gaze points, fixation data, and blink patterns recorded from participants.

o Usage: Train and validate gaze prediction models, analyze AOIs.

o Cleaned dataset prepared for accuracy testing and visualization.

► Key Features

• Gaze Data Collection: Real-time webcam-based eye tracking via MediaPipe FaceMesh.
 
• Behavioural Metric Extraction: Fixations, saccades, blink frequency, pursuit gain, gaze variability. 

• ASD Risk Prediction: Calibrated ensemble ML model (Low / Borderline / High risk). 

• Real-time HUD: Live gaze overlays during screening.

<img width="1113" height="1034" alt="image" src="https://github.com/user-attachments/assets/248caf71-3543-451e-9fd4-e71e5cf132ee" />


►User Interface Development:

The frontend is implemented using React.js with a minimalistic, distraction-free layout designed for real-time tracking tasks. 
Key features include: 
• Live video preview with gaze overlays. 
• Fixation cross, pursuit target, and saccade stimuli presentation.
• Session progress and status indicators. 
• Real-time metrics such as blink count, fixation time, and gaze direction. 
• Final ASD risk scoring display with interpretation.

►Stimuli Rendering Module:

This module renders dynamically controlled tasks including:

• Central fixation targets, Horizontal/vertical smooth pursuit paths.

• Randomized saccade jumping points.

• Social and non-social video stimuli

<img width="1608" height="897" alt="image" src="https://github.com/user-attachments/assets/4b97e85e-6af6-487b-96ac-422847a9c0f9" />

<img width="1553" height="874" alt="image" src="https://github.com/user-attachments/assets/ed1b166a-5811-400f-84e8-41bed9d2b397" />

<img width="2410" height="890" alt="image" src="https://github.com/user-attachments/assets/cdfaf2c7-4b6d-48c9-8514-79d8fe4154d1" />

►Backend Implementation:

The backend is implemented in Python using FastAPI/Flask. It manages request handling, real-time landmark extraction, and ML inference.

►Backend Modules 

Frame Receiver: Accepts frames streamed from frontend, Gaze Tracking Engine: Extracts landmarks using MediaPipe FaceMesh, Behavioural Feature Extractor: Computes fixations, saccades, blinks, pursuit gain, Model Inference Engine: Runs ensemble classifier on extracted features and Report Generator: Produces structured output artifacts.

►Behavioural Feature Extraction:

 The behavioural features computed by the system are essential for differentiating ASD and non-ASD patterns. The feature pipeline includes:
 
• Fixation Metrics: Count, mean duration, stability. 
• Saccadic Metrics: Frequency, amplitude, velocity. 
• Blink Metrics: Blink count, blink intervals, blink irregularities. 
• Smooth Pursuit Metrics: Tracking accuracy, pursuit gain, lag error.
• Gaze Variability Metrics: Gaze dispersion, jitter, cluster spread

<img width="3000" height="755" alt="image" src="https://github.com/user-attachments/assets/b847f694-2c93-4611-9481-cde1c91dfe5b" />

This project presented a complete, AI-powered eye-tracking system designed to support early-stage screening of Autism Spectrum Disorder (ASD) using accessible, webcam-based technology. 

Designed for clinics, schools, and home settings, it combines real-time processing, accessible hardware, and structured reporting. Looking at how often focus changes, size of eye motions, blinking speed, or ability to follow objects gives doctors clearer clues, Research constantly ties these actions to ASD. 

The system runs videos at 30 frames per second, displaying real-time motion while flagging dangerous situations which helps teachers or healthcare workers catch events instantly – skipping slow playback check.

Auto data collection works well with regular check-ups on features, cutting 4 down hands-on work – great for research, student wellness drives, or trial runs in clinics

<img width="6477" height="388" alt="image" src="https://github.com/user-attachments/assets/4f47fa5b-9c85-4993-a8b9-5b1cd272ea0e" />

► Future Enhancements:

• Deep Learning–Based Gaze Models: Incorporate transformer or CNN-based gaze-estimation models to improve accuracy under challenging environments.

 • Multimodal Behavioural Signals: Integrate voice patterns, gesture recognition, and micro-expressions to support multimodal ASD analysis.

 • Adaptive Testing Protocols: Develop dynamic stimuli that adapt to user behaviour in real time to improve engagement and reduce noisy samples. 

• Cross-Device Calibration: Implement automated calibration methods for different webcams and screen resolutions. 

• Clinical Validation Studies: Conduct large-scale hospital trials and cross cultural studies to assess generalizability and long- term reliability. 

<img width="3015" height="440" alt="image" src="https://github.com/user-attachments/assets/ba6c0aef-be0f-46bb-9c73-c6bcab32b15d" />











