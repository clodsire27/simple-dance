# Simple Dance Pose Comparison System

This project is a course project that implements a real-time human pose comparison system using MediaPipe Pose and OpenCV.  
The system compares a user's live webcam pose with a reference dance video and computes a similarity score based on keypoint distances and joint angle differences.

## Demo

![Pose Comparison Demo](gg.GIF)

## Overview

The goal of this project is to analyze and evaluate human motion similarity by extracting skeletal pose information from both a reference video and real-time webcam input.  
Pose similarity is quantified using a combination of weighted keypoint distance and joint angle error metrics.

## Features

- Extracts pose keypoints from a reference dance video  
- Saves reference pose data in JSON format  
- Captures real-time pose data from a webcam  
- Visualizes skeletons for both reference video and user input  
- Computes pose similarity using:
  - Weighted keypoint distance
  - Joint angle differences  
- Displays a real-time similarity score  
- Outputs a final similarity score after execution  

## Technologies Used

- Python 3  
- OpenCV (cv2)  
- MediaPipe  
- NumPy  
- JSON  

## Project Workflow

### 1. Reference Pose Extraction
- Reads a reference dance video  
- Extracts pose landmarks using MediaPipe Pose  
- Stores keypoints for each frame in `reference_keypoints.json`

### 2. Real-Time Pose Capture
- Captures video input from a webcam  
- Extracts pose landmarks in real time  

### 3. Pose Comparison
- Compares user pose with reference pose using:
  - Euclidean distance between corresponding keypoints  
  - Angle differences at major joints  
- Applies higher weights to important body parts such as arms, legs, and hips  

### 4. Scoring
- Computes a real-time similarity score ranging from 0 to 100  
- Displays the score on the video output  
- Outputs a final similarity score when execution ends  

## File Structure

