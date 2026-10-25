# PySPRESSO :coffee:
**PySPRESSO** - **Py**thon **S**tatistical **P**rocessing and **RE**porting for **S**cientific **S**tudies in **O**mics. 
##
> *It is a comprehensive tool designed to streamline workflows for processing and reporting peak matrix data, especially for applications in metabolomics or other omics fields.*
>
> ### Features
> * **Data Filtering:** Handles missing values, removes outliers, and corrects for blank intensity.
> * **Batch Correction:** Corrects batch effects using QC samples.
> * **Normalization:** Supports methods like z-scores and Probabilistic Quotient Normalization (PQN).
> * **Statistical Analysis:** Implements PCA and PLS-DA.
> * **Report Generation:** Automatically generates PDF reports for the processed data.
##

> [!IMPORTANT]
> *This is a WIP (Work in progress)* APP. If you want to add some feature feel free to do so (or contact me if you don't code yourself. :innocent: )

> [!TIP]
> ## DEMO
> If you wanna try the functionality
> 
> https://colab.research.google.com/github/Najlaron/PySPRESSO/blob/main/demo-PySPRESSO-pipeline.ipynb

## Build and run the desktop app (`.exe`)

### Requirements
- [Python](https://www.python.org/downloads/)
- [Node.js](https://nodejs.org/en/download)

### 1) Build the app
From the project root, run:

`build-desktop.bat`

This script:
- builds the frontend
- packages the app with PyInstaller
- creates the output in `pyspresso\dist\PySPRESSO\`

Rebuilding replaces the entire `dist\PySPRESSO` folder (old `.exe` and bundled libraries are deleted and recreated).

### 2) Start the app
Open the built `PySPRESSO` folder, then double-click `PySPRESSO.exe`.

A console window appears — keep it open while using the app.  
Your browser should open automatically at `http://127.0.0.1:5000`.

If the browser does not open automatically, open that URL manually.

### 3) Use the app
Work with PySPRESSO in the browser window normally.

As long as the console window stays open, the backend is running.

### 4) Stop the app
To stop PySPRESSO, close the console window opened by `PySPRESSO.exe`.

### Where your data is stored (important)
Desktop app data is stored in `%LOCALAPPDATA%\PySPRESSO`.

This includes:
- uploaded files
- generated outputs
- local database

Because this data is stored outside the app folder, you can rebuild or replace the `.exe` package without losing your data.