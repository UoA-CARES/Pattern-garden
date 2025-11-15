
# **Pattern Garden**

**Adaptive Cognitive-Training Framework for the iRobi Socially Assistive Robot**

Pattern Garden is an adaptive, procedurally generated cognitive-training system designed for **older adults with amnestic Mild Cognitive Impairment (aMCI) or early-stage Alzheimer’s disease (AD)**.  
It runs on a **Linux-upgraded iRobi robot**, using Kivy/KivyMD for a dementia-friendly touchscreen interface and an IRT-based adaptive engine that personalises difficulty in real time.

----------

## ✨ Features

### **Adaptive Cognitive Training (IRT-Based)**

-   Bayesian θ-updates after every trial
    
-   Dynamic difficulty scaling
    
-   Uses probe trials, anchor repeats, and zero-shot generalisation
    
-   Rolling accuracy window to decide phase progression or regression
    

### **Procedurally Generated Visual Stimuli**

-   L-systems
    
-   Perlin/Worley noise
    
-   Shape primitives
    
-   Rotations, occlusions, distractors, flipping, jitter
    

### **Dementia-Friendly UI**

-   Large tap targets
    
-   Warm colours and reduced visual clutter
    
-   Slow-paced transitions
    
-   Clear spoken instructions via offline text-to-speech
    
-   Break/resume functionality
    

### **Full Session Logging (Stealth Assessment)**

Automatically exports a CSV containing:

-   Accuracy
    
-   Reaction time
    
-   θ estimates
    
-   Difficulty level
    
-   Probe/anchor flags
    
-   Time stamps
    
-   Item metadata
    

### **iRobi Integration**

-   Motor gestures (“dance” feedback)
    
-   Facial LED changes
    
-   Offline TTS for instructions
    
-   Runs fully on a portable power bank
    

----------

## 📁 Repository Structure

`Pattern-garden/
├── game.py # Main KivyMD application ├── engine.py # Adaptive IRT engine ├── stimuli.py # Procedural image generators + transforms ├── models.py # Item definitions + difficulty parameters ├── ui/
│   └── widgets.py # Custom widgets (e.g., ImageButton) ├── config.py # UI constants & dementia-friendly settings ├── results/ # Auto-generated participant CSVs └── README.md` 

----------

## 🚀 Installation

### **1. Clone the repository**

`git clone https://github.com/UoA-CARES/Pattern-garden.git cd Pattern-garden` 

### **2. Create a virtual environment**

`python3 -m venv venv source venv/bin/activate` 

### **3. Install dependencies**

`pip install -r requirements.txt` 

### **4. Run the application**

`python game.py` 

----------

## 🧠 Adaptive Engine Overview

Pattern Garden uses a lightweight Item Response Theory model:

`p(correct) = expit(a * (θ - b))` 

-   θ updates after every non-probe trial
    
-   Expected Information guides item selection
    
-   Difficulty tuning uses a moving accuracy window
    
-   Automatic phase progression:
    

`Warm-up 2AFC → Sample 2AFC → 3AFC → 4AFC → Grid  2×2 → Grid  3×3 ↑                       ↓                     ↓
   accuracy < 40% ←––––– performance ≥ 70% –––––→ next phase` 

----------

## 🖼 Procedural Stimuli

All images are generated at runtime — nothing is pre-stored.

**Stimulus families include:**

-   L-system fractals
    
-   Perlin/Worley noise
    
-   Geometric shapes
    
-   Mixed transformations (rotation, occlusion, distractor overlays)
    

**Foil generation includes:**

-   Seed jitter
    
-   Random flips
    
-   Randomized transformations
    
-   Distractor strokes or overlays
    

This ensures infinite replayability and eliminates memory bias.

----------

## 📊 Data Output

Each session automatically saves a CSV using:

`P###_results.csv` 

### **Logged Variables**

Variable

Description

`trial_index`

Trial number

`phase`

Current difficulty stage

`correct`

Response accuracy

`reaction_time_s`

Response latency

`theta_after`

Updated latent ability

`probe` / `anchor`

Stealth assessment flags

`difficulty_level`

Dynamic difficulty

`item_a`, `item_b`

IRT parameters

`timestamp_end`

ISO 8601 timestamp

Used for:

-   Learning curves
    
-   θ trajectories
    
-   Robustness testing (rotation/occlusion)
    
-   Participant summaries
    

----------

## 🤖 iRobi Integration

Pattern Garden runs on a **Linux-based Raspberry Pi installed inside iRobi**, enabling:

-   Python-based motor control
    
-   Facial LED cues
    
-   Clear, slow-paced spoken instructions
    
-   Positive reinforcement via “robot dance”
    
-   Full portability using a power bank
    

> **Note:** Occasional motor “buzzing” may occur due to power-bank voltage throttling.

----------

## 🔬 Research Context

Pattern Garden was developed as part of a Master’s project at the **University of Auckland (CARES Research Group)** focusing on:

-   Early cognitive decline
    
-   Adaptive psychometrics
    
-   Socially assistive robotics
    
-   Gamified rehabilitation
    

**Ethics approval:** `UAHPEC29819`  
**Participants:** P001–P005 (pilot evaluation)  
A full literature survey is included in `lit_survey.docx`.

----------

## 🛣 Roadmap

-   Add emotion-aware adaptation
    
-   Add caregiver dashboard
    
-   Expand the item bank (logic, planning, semantic tasks)
    
-   Add multi-step tasks (n-back, sequencing, route planning)
    
-   Integrate multimodal sensing (vision, speech, behaviour)
    

----------

## 👩‍💻 Author

**Zahra Ally**  
Master of Robotics & Automation  
University of Auckland — CARES Research Group
