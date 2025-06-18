# agent.md

## 👤 Who I Am

I am a Senior Data Scientist working in the field of methane emissions monitoring, particularly on oil and gas sites. My work involves developing advanced algorithms for source detection, localization, and quantification using open-path CH₄ laser sensor data. I lead a small data science team and work extensively with probabilistic models, time series analysis, and geospatial uncertainty evaluation.

## 🧠 My Expertise

- Gas modeling using **Gaussian plume physics**
- **Bayesian inference** and **RJMCMC** for source estimation (via `pyELQ`)
- Temporal and spatial uncertainty analysis using **Shannon entropy**, **PCA**, and **ellipse overlap metrics**
- Real-time methane sensor data from **retroreflector arrays**, with sequential sampling and time-synchronized aggregation
- Anomaly detection using **Isolation Forest**, rolling statistics, and spike windowing
- Heavy use of **Python**, **Pandas**, **xarray**, **Matplotlib**, **GeoPandas**, and **Scikit-learn**
- Database work with **PostgreSQL** on **WSL** and **Azure**
- Workflow automation using **Mamba**, **VS Code**, and Git
- Familiarity with **GIS**, **DBSCAN**, and **KDE** for spatial inference

## 🧰 Tools I Use

- Python (Mamba/Conda environments)
- VS Code + GitHub Copilot
- Azure Blob + PostgreSQL (for sensor data storage and retrieval)
- pyELQ (internal Gaussian plume modeling package)
- Plotly, matplotlib, seaborn (visualization)
- pandas, numpy, xarray (data wrangling)
- geopandas, shapely, cartopy (mapping)
- scikit-learn (ML/detection)
- Jupyter/VSCode notebooks

## 🧭 What I Want Codex To Help With

- Writing and debugging **Python scripts** for methane event detection, inverse modeling, or geospatial clustering
- Wrapping or extending pyELQ features for new use cases
- Automating pipelines to convert raw sensor data into quantification-ready formats
- Improving **performance** of multi-run MCMC workflows
- Building or debugging **dashboards** that query Azure Blob data
- Suggesting alternatives to CALPUFF and Gaussian plume models in Python
- Visualizing **temporal/spatial detection confidence**

## 🧱 Coding Preferences

- Modular, reusable Python functions
- Minimal but clear comments
- Pandas/xarray-first mindset for structured data
- Prefer clarity over excessive abstraction
- Avoids magic strings or hard-coded values — uses config patterns

## 🧪 Sample Tasks

- Calculate ellipse overlaps from posterior XYZ samples
- Apply entropy metric to emission source quantification runs
- Bin and align sensor time series for correlation analysis
- Generate interactive map of CH₄ hotspots from clustering
- Build a detection windowing model that adjusts based on rolling mean, std, and rate of change

