"""AI coach: turns the model output into a personalised micro-habit energy plan.

Uses Google Gemini (google-genai SDK). If no API key is available or the call
fails, a rule-based fallback plan is returned so the app still works offline.
"""
import os

try:
    from dotenv import load_dotenv

    load_dotenv()  # reads GEMINI_API_KEY from a local .env file if present
except ImportError:
    pass

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def build_prompt(profile: dict, result: dict) -> str:
    return f"""You are a friendly, evidence-based student productivity coach.

A student's day looks like this:
- Sleep: {profile['sleep_hours']} h (quality {profile['sleep_quality']}/10)
- Screen time: {profile['screen_time_hours']} h
- Caffeine: {profile['caffeine_cups']} cups
- Morning mood: {profile['morning_mood']}/10
- Planned study time: {profile['study_hours_planned']} h

Model predictions:
- Productivity score: {result['productivity']:.0f}/100
- Deep work possible: about {result['deep_work_hours']} hours
- Burnout probability: {result['burnout']:.0f}% ({result['burnout_label']})
- Predicted focus PEAK windows: {', '.join(result['peak_ranges']) or 'none'}
- Predicted focus SLUMP windows: {', '.join(result['slump_ranges']) or 'none'}

Write a short personalised energy plan with 4-6 micro-habits. Rules:
- Each habit must be tied to a specific clock time and say WHY, e.g.
  "Take a 15-minute walk at 2:00 PM because your focus dips then."
- Put the hardest study work inside the peak windows.
- Put light tasks, breaks, water, sunlight or short walks in slump windows.
- If burnout is moderate/high, include one recovery habit for tonight.
- Mention caffeine timing or screen-time advice only if it is relevant.
- Keep it under 200 words, use a bulleted list, warm and non-judgemental tone.
"""


def fallback_plan(profile: dict, result: dict) -> str:
    lines = []
    if result["peak_ranges"]:
        lines.append(
            f"- **{result['peak_ranges'][0]}** - your best focus window. "
            "Do your hardest subject here with phone in another room."
        )
    for window in result["slump_ranges"][:2]:
        lines.append(
            f"- **{window}** - focus dips. Take a 15-minute walk, drink water, "
            "and do light revision or admin tasks."
        )
    if profile["caffeine_cups"] >= 3:
        lines.append(
            "- Skip caffeine after **2:00 PM** so it does not hurt tonight's sleep."
        )
    if profile["screen_time_hours"] > 7:
        lines.append(
            "- Use the 20-20-20 rule: every 20 minutes look 20 feet away for 20 seconds."
        )
    if result["burnout"] >= 35:
        lines.append(
            "- Tonight: screens off 45 minutes before bed and aim for "
            f"{max(profile['sleep_hours'] + 1, 7.5):.1f} hours of sleep."
        )
    return "\n".join(lines)


def generate_plan(profile: dict, result: dict, api_key: str | None = None):
    """Return (plan_text, source) where source is 'gemini' or 'fallback'."""
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        return fallback_plan(profile, result), "fallback (no API key)"
    try:
        from google import genai

        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model=DEFAULT_MODEL, contents=build_prompt(profile, result)
        )
        text = (response.text or "").strip()
        if not text:
            raise ValueError("empty response")
        return text, "gemini"
    except Exception as exc:  # network, quota, bad key, model name...
        return fallback_plan(profile, result), f"fallback ({type(exc).__name__})"
