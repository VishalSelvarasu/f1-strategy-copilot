# F1 Strategy Copilot

Predicting likely F1 pit windows from FastF1 lap data.

## Pit-window prediction

This project is a first prototype of a race-strategy support tool.

The current task is simple: given the current lap context for a driver, predict whether that driver is likely to pit within the next 3 laps.

## Race data

The dataset is built from 5 races from the 2024 season using FastF1:

- Bahrain
- Saudi Arabia
- Australia
- Japan
- China

Current dataset size:
- 4,878 lap rows
- 541 positive `NextPitIn3Laps` labels

## Current baseline

The current model is a Random Forest classifier trained on lap-level features such as:
- lap number
- tyre life
- stint
- position
- lap time
- sector times
- compound
- track status

Current result on the held-out split:
- Accuracy: 0.91
- Precision (pit-window class): 0.56
- Recall (pit-window class): 0.72
- F1-score (pit-window class): 0.63

At this stage, I’m treating it as a decision-support baseline rather than a final strategy model.

## Dashboard

The repo also includes a small Streamlit app for exploring predictions by race and driver.

Current dashboard flow:
- select a race
- select a driver
- inspect predicted pit probability
- view high-risk pit-window laps

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python .\src\data_pipeline.py
python .\src\model.py
python -m streamlit run .\app\streamlit_app.py
```

## Next steps

- add more races and seasons
- improve degradation-related features
- tune the prediction threshold more systematically
- make the dashboard more informative
- add a better explanation layer for model decisions