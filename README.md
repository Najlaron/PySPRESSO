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

## Running the app

### Option A: Desktop app (recommended for end users)
No Docker, Node.js or command line knowledge required.

1. Download/build `PySPRESSO.exe` (see "Building the desktop app" below).
2. Double-click `PySPRESSO.exe` inside the `PySPRESSO` folder.
3. A console window opens (keep it open) and your browser opens automatically at `http://127.0.0.1:5000`.
4. To stop the app, close the console window.

Your data (uploaded files, generated outputs, database) is stored in `%LOCALAPPDATA%\PySPRESSO`, separate from the app folder — this is what makes it safe to rebuild/replace `PySPRESSO.exe` (see below) without losing your data. Don't run the desktop app and Docker at the same time — both use port 5000, and whichever started first will silently receive all requests.

#### Building the desktop app
Requires Python and Node.js installed once, only for building:
```
build-desktop.bat
```
This builds the frontend and bundles the whole app with PyInstaller into `pyspresso\dist\PySPRESSO\`. **Rebuilding replaces that entire folder** (old `.exe`/libraries are deleted and rewritten) — but since your data lives in `%LOCALAPPDATA%\PySPRESSO` instead, it's untouched by rebuilds. Copy the `dist\PySPRESSO` folder wherever you like and share/run `PySPRESSO.exe` from inside it.

### Option B: Docker (for development)
```
docker compose up -d
```
Then open `http://localhost:5173`. Use `start-app.bat` / `stop-app.bat` on Windows as shortcuts.

### Backup and restore (Windows desktop app)
Desktop app data is stored in `%LOCALAPPDATA%\PySPRESSO`.

- Create backup zip: `backup-data.bat`
- Restore from zip: `restore-data.bat "path-to-backup-zip"`

The backup zip contains `instance`, `outputs`, and `uploads`.
After rebuilding the `.exe`, restoring that zip will put those folders back into `%LOCALAPPDATA%\PySPRESSO`.
