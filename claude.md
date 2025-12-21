# ArcFace Face Recognition Project - Session Notes

**Date:** 2025-12-21
**Container:** jetson-arcface:v1 (Docker)

---

## Project Overview

This is a facial recognition system running on NVIDIA Jetson hardware using:
- **YOLO11-Nano** (`yolo11n.pt`) - Person/object detection
- **ArcFace** (`arcface.onnx`) - Face embedding extraction (512D vectors)
- **Encrypted storage** - Fernet encryption for embeddings database

**Main Scripts:**
- `enroll.py` - Enroll new people with images
- `identify_live.py` - Real-time identification from camera
- `verify_live.py` - Real-time verification from camera
- `arcface_infer.py` - ArcFace inference engine
- `crypto_store.py` - Encrypted embedding storage/retrieval

---

## Problem Reported

**Issue:** System always identifies people as "Indraneel" regardless of who's in front of the camera.

---

## What I Did

### 1. Initial Investigation

**Analyzed stored embeddings:**
```
Indraneel: shape=(512,), norm=1.000000
Pramod:    shape=(512,), norm=1.000000
Cosine similarity: 0.8997 (89.97% similar!)
Distance: 0.100272
```

**Root Cause:** The two enrolled embeddings were almost identical (distance 0.100272, barely above threshold 0.1). This meant the system couldn't distinguish between them.

### 2. Cleared and Re-enrolled

**Actions:**
- Cleared `embeddings.json` (set to empty `{}`)
- Re-enrolled with better images:
  - `Pramod.jpeg` from `/home/user/images/`
  - `Indraneel.jpeg` from `/home/user/images/`

**New Results:**
```
Pramod:    shape=(512,), norm=1.000000 ✓
Indraneel: shape=(512,), norm=1.000000 ✓
Cosine similarity: 0.2845
Distance: 0.7155 ✓ GOOD - Clearly distinguishable
```

**Improvement:** Distance increased from 0.1003 to 0.7155 (7x better separation!)

### 3. Tested Live Identification

**Command:** `python -u identify_live.py` (running in background, task ID: bfbe132)

**Results:**
- Unknown person: `✗ UNKNOWN (closest: Pramod, distance: 0.96-0.98)` ✓ Working correctly
- Enrolled people (Pramod/Indraneel): Still showing as UNKNOWN with high distances (0.83-0.98)

---

## Current Problem: YOLO Person Detection vs Face Detection

### Root Cause

**The fundamental issue:** YOLO11 is detecting **"person" class (full body)**, not faces.

**Enrollment Flow:**
```
Image → YOLO person detection → Full body crop → ArcFace → Embedding
```

**Live Identification Flow:**
```
Camera → YOLO person detection → Full body crop → ArcFace → Embedding
```

**Problem:**
- ArcFace is designed for **face recognition**, not full-body recognition
- Different body poses/angles produce completely different embeddings
- Even the same person in different poses gets distance ~0.85-0.98 (vs threshold 0.1)

### Why Re-enrollment Helped (but didn't fully fix)

- New enrollment images created more distinct full-body embeddings
- Pramod and Indraneel are now distinguishable from each other (0.7155 distance)
- BUT live detection still fails because body pose changes too much

---

## What Needs to Be Done

### **CRITICAL FIX: Switch from Person Detection to Face Detection**

#### Option 1: Use YOLO Face Detection Model (Recommended)

Replace `yolo11n.pt` with a face-specific YOLO model:

1. Download YOLO face detection model
2. Update `enroll.py` and `identify_live.py` to use face model
3. Re-enroll both people with new face crops
4. Test live identification

**Files to modify:**
- `enroll.py:20` - Change `YOLO("yolo11n.pt")` to face model
- `identify_live.py:20` - Change `YOLO("yolo11n.pt")` to face model
- Both scripts: Change class filter from `cls == 0` (person) to appropriate face class

#### Option 2: Add Face Detection After Person Detection

Use a cascade approach:
```
Camera → YOLO (person) → Crop person → Face detector (e.g., Haar/DNN) → Face crop → ArcFace
```

**Implementation:**
1. Add OpenCV face detector (e.g., `cv2.CascadeClassifier` or DNN-based)
2. After YOLO person detection, run face detector on person crop
3. Extract face region from person bounding box
4. Send face crop to ArcFace

**Pros:** Works with existing YOLO model
**Cons:** Two-stage detection is slower

#### Option 3: Use MediaPipe Face Detection

Replace YOLO with MediaPipe Face Detection:
```python
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
```

**Pros:** Fast, accurate, designed for faces
**Cons:** Requires MediaPipe installation and code rewrite

---

## Recommended Next Steps

### Immediate (Quick Fix - Option 2)

**Modify enrollment and identification to use OpenCV face detection:**

1. **Update `enroll.py`:**
   ```python
   # After YOLO person detection (line 28)
   x1, y1, x2, y2 = boxes[0].astype(int)
   person_crop = img[y1:y2, x1:x2]

   # Add face detection on person crop
   face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
   gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
   faces = face_cascade.detectMultiScale(gray, 1.1, 4)

   if len(faces) > 0:
       fx, fy, fw, fh = faces[0]
       face_crop = person_crop[fy:fy+fh, fx:fx+fw]
   else:
       print("No face detected in person crop")
       sys.exit(1)
   ```

2. **Update `identify_live.py`:** Same face detection logic in the detection loop (after line 46)

3. **Clear and re-enroll** with face detection enabled

4. **Test again**

### Long-term (Better Solution - Option 1)

1. Find and download a YOLO face detection model (e.g., YOLOv8-face)
2. Replace `yolo11n.pt` with face model
3. Update class filtering in both scripts
4. Re-enroll and test

---

## Current State

**Database:** 2 enrolled people (Pramod, Indraneel)
**Enrollment Quality:** GOOD (distance 0.7155 between them)
**Live Identification:** NOT WORKING - detects enrolled people as UNKNOWN due to YOLO person detection issue

**Background Process:** `identify_live.py` running (task ID: bfbe132)
**Log file:** `/tmp/claude/-home-user-arcface-project/tasks/bfbe132.output`

**To kill background process:**
```bash
pkill -f identify_live.py
```

---

## Code Issues Found

**File:** `crypto_store.py`
**Issue:** Duplicate function definition for `load_all_embeddings()` (lines 39-47 and 49-67)
**Impact:** Low - second definition overrides first, both are functionally identical
**Recommendation:** Remove the first definition (lines 39-47)

---

## Hardware Info

- **Platform:** NVIDIA Jetson Orin
- **OS:** Ubuntu 22.04.3 LTS
- **CUDA:** 12.6
- **TensorRT:** 10.3.0
- **Current Execution:** CPU (CPUExecutionProvider in `arcface_infer.py:10`)
- **Potential:** Could enable GPU acceleration by changing to CUDAExecutionProvider

---

## Commands Reference

**Enroll a person:**
```bash
python enroll.py "PersonName" /path/to/image.jpg
```

**Start live identification:**
```bash
python identify_live.py
```

**Check enrolled people:**
```bash
python3 -c "from crypto_store import load_all_embeddings; import numpy as np; embs = load_all_embeddings(); print(f'Enrolled: {list(embs.keys())}'); names = list(embs.keys()); print(f'Distance: {1 - np.dot(embs[names[0]], embs[names[1]]):.4f}') if len(names) == 2 else None"
```

**Clear database:**
```bash
echo '{}' > embeddings.json
```

---

## File Locations

```
/home/user/arcface_project/
├── arcface.onnx              # ArcFace model (131 MB)
├── yolo11n.pt                # YOLO11-Nano model (5.4 MB)
├── arcface_infer.py          # ArcFace inference class
├── enroll.py                 # Enrollment script
├── identify_live.py          # Live identification script
├── verify_live.py            # Live verification script
├── crypto_store.py           # Encrypted storage
├── embeddings.json           # Encrypted embeddings database
├── secret.key                # Fernet encryption key (44 bytes)
└── claude.md                 # This file

/home/user/images/
├── Pramod.jpeg               # Enrollment photo
└── Indraneel.jpeg            # Enrollment photo
```

---

## Next Session TODO

- [ ] Implement face detection (Option 2 from recommendations)
- [ ] Clear and re-enroll with face detection enabled
- [ ] Test live identification with enrolled people
- [ ] If working well, clean up duplicate function in `crypto_store.py`
- [ ] Consider GPU acceleration for better performance
- [ ] Add visual display to `identify_live.py` (show camera feed with bounding boxes)

---

## Questions to Consider

1. Do you want face-only detection or person+face cascade?
2. Should we add visual display (camera feed with boxes/names)?
3. Should we enable GPU acceleration for faster inference?
4. What should happen when multiple people are in frame?
5. Do you want confidence scores displayed along with identification?
