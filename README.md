# 🧠 Deep Work & Burnout Productivity Predictor

A college mini project that combines **sleep and study habits** to predict how
productive your day will be, how much **deep work** you can realistically do,
and your **burnout probability**, then asks **Google Gemini** to write a
personalised micro-habit energy plan.

## Features
- **Linear Regression** (scikit-learn) predicts a Productivity Score (0-100) and a Burnout Probability (0-100 %).
- Deep Work Hours are derived from the productivity score (max 6 h/day).
- Hour-by-hour **focus curve** with detected **peak** and **slump** windows.
- **Gemini AI coach** writes habits like *"Take a 15-minute walk at 2:00 PM because your focus dips then."*
- Offline **rule-based fallback** if no API key is provided.
- Clean **Streamlit** dashboard.

## Inputs
Sleep duration, sleep quality, screen time, caffeine intake, morning mood, planned study hours.

## Project structure
```
.
├── app.py              # Streamlit UI
├── predictor.py        # loads models, runs predictions
├── focus_curve.py      # hourly focus curve + slump/peak detection
├── ai_coach.py         # Gemini integration + fallback plan
├── generate_data.py    # synthetic dataset generator
├── train_model.py      # trains the two Linear Regression models
├── requirements.txt
├── .env.example
└── data/ , models/     # generated files
```

## Setup
```bash
git clone https://github.com/<your-username>/deep-work-burnout-predictor.git
cd deep-work-burnout-predictor

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python generate_data.py          # creates data/student_productivity.csv
python train_model.py            # trains and saves models/

cp .env.example .env             # add your Gemini API key (optional)
streamlit run app.py
```
You can also paste the API key into the app sidebar instead of using `.env`.

## How it works
1. `generate_data.py` simulates 2000 student-days using realistic relationships
   (more sleep/mood = better; more screen time/caffeine/study load = worse).
2. `train_model.py` fits two `LinearRegression` models with an 80/20 split and reports R² and MAE.
3. `focus_curve.py` builds a circadian focus curve scaled by the prediction and
   adjusted for sleep debt and caffeine, then finds slump and peak hours.
4. `ai_coach.py` sends the numbers and the slump/peak windows to Gemini and
   shows the generated plan.

## Limitations
- Training data is **synthetic**; for a stronger project, collect a real survey/dataset and retrain.
- Linear Regression assumes linear effects, which is a simplification (e.g. too much sleep can also reduce productivity).
- Not medical advice.

## Future work
Real survey data, Random Forest / Gradient Boosting comparison, daily log tracking, wearable data import.

## License
MIT
