
# PWST-YOLOv7-Mini-Trash-Dataset

This repository contains the **PWST-YOLOv7** code and the Mini-Trash dataset used in our paper, which is currently under minor revision for the *IEEE Journal of Oceanic Engineering (JOE)*.

## Table of Contents

- [Introduction](#introduction)
- [Datasets](#datasets)
- [Pre-trained Weights](#pre-trained-weights)
- [Dataset Structure](#dataset-structure)
- [Citation](#citation)

---

## Introduction

This project provides open access to our custom Mini-Trash dataset and associated YOLOv7 models for underwater trash detection. Both the dataset and code are intended for academic and non-commercial use.

---

## Datasets

- **Mini-Trash Dataset**  
  *The dataset created and used in the PWST-YOLOv7 paper*  
  [Download Link](https://drive.google.com/file/d/1U1b-TxiKt6ug3hq_tWohX3pILj45QrmR/view?usp=drive_link)

- **TrashCan Dataset (YOLO format)**  
  *Original TrashCan dataset, transformed to YOLO format*  
  [Download Link](https://drive.google.com/file/d/1n957_9mqipm7uBjQgtCaMk8JpsJT624R/view?usp=drive_link)

---

## Pre-trained Weights

- **YOLOv7 Baseline (TrashCan Dataset)**  
  [Download](https://drive.google.com/file/d/1hwrDN7miv_XTjPGIa0TXZhIevmwdIN9d/view?usp=drive_link)

- **PWST-YOLOv7 (TrashCan Dataset)**  
  [Download](https://drive.google.com/file/d/1yo8BE50DF5xY8qfWSVezh6-U1oMvtk8K/view?usp=drive_link)

---

## Dataset Structure

Both datasets are provided in YOLO format, organized as follows:
```plaintext
yolo_all_cls/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
>>>>>>> 4745dee0821e8f4612c6c20f9cce45b8c207e6fd
