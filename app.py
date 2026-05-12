"""
AI Personalized Diet & Fitness Recommendation System
----------------------------------------------------
Run locally with:
    streamlit run app.py
"""

import os
import random

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# Page configuration (must be the first Streamlit command)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Diet Planner",
    layout="centered",
    page_icon="🥗",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# Custom CSS — high-contrast, startup-style look
# Key rules:
#   * Soft light-blue → light-purple background gradient
#   * Pure white cards with visible borders + shadows
#   * No backdrop-filter (causes interactivity bugs with sliders)
#   * No global <hr> overrides (can break slider track rendering)
#   * No pointer-events overrides; sliders retain native behavior
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #111827;
}

/* ---- Soft gradient background ---- */
.stApp {
    background: linear-gradient(135deg, #e0ecff 0%, #eef2ff 45%, #ede9fe 100%);
    background-attachment: fixed;
}

/* ---- Container spacing ---- */
.main .block-container {
    padding-top: 2.5rem;
    padding-bottom: 4rem;
    max-width: 960px;
}

/* ---- Hide Streamlit default chrome ---- */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}

/* ---- Hero ---- */
.hero-wrap {
    text-align: center;
    margin-bottom: 1.8rem;
}
.hero-title {
    font-size: 2.7rem;
    font-weight: 800;
    background: linear-gradient(135deg, #4338ca 0%, #7c3aed 50%, #db2777 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 0.35rem;
    letter-spacing: -0.025em;
    line-height: 1.15;
}
.hero-subtitle {
    text-align: center;
    color: #374151;
    font-size: 1.05rem;
    margin-bottom: 1rem;
    font-weight: 400;
    max-width: 640px;
    margin-left: auto;
    margin-right: auto;
}
.hero-badge {
    display: inline-block;
    padding: 0.35rem 0.9rem;
    background: #ffffff;
    color: #4338ca;
    border: 1px solid #c7d2fe;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    box-shadow: 0 2px 8px rgba(79, 70, 229, 0.12);
}

/* ---- Section titles ---- */
.section-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #111827;
    margin: 0.2rem 0 0.6rem 0;
    letter-spacing: -0.01em;
}
.section-caption {
    color: #4b5563;
    font-size: 0.92rem;
    margin: 0 0 1.2rem 0;
}

/* ---- Card containers (st.container(border=True)) ---- */
/* Pure white, clearly visible border, real shadow */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff !important;
    border-radius: 18px !important;
    border: 1px solid #e5e7eb !important;
    box-shadow: 0 6px 22px rgba(17, 24, 39, 0.08), 0 2px 6px rgba(17, 24, 39, 0.04);
}

/* ---- Text inputs & number inputs ---- */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: #ffffff !important;
    color: #111827 !important;
    border-radius: 10px !important;
    border: 1px solid #d1d5db !important;
    padding: 0.55rem 0.8rem !important;
    font-size: 0.98rem !important;
}
.stTextInput > div > div > input::placeholder {
    color: #9ca3af !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.18) !important;
}

/* ---- Labels ---- */
.stTextInput label,
.stNumberInput label,
.stSelectbox label,
.stRadio label,
.stSlider label,
.stCheckbox label {
    font-weight: 600 !important;
    color: #1f2937 !important;
    font-size: 0.92rem !important;
}

/* ---- Selectbox ---- */
div[data-baseweb="select"] > div {
    background: #ffffff !important;
    border-radius: 10px !important;
    border-color: #d1d5db !important;
    color: #111827 !important;
}

/* ---- Primary CTA button ---- */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4338ca 0%, #7c3aed 55%, #db2777 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.95rem 1.5rem !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em;
    box-shadow: 0 10px 22px rgba(124, 58, 237, 0.40);
    transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease;
    width: 100%;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 14px 28px rgba(124, 58, 237, 0.50);
    filter: brightness(1.05);
}
.stButton > button[kind="primary"]:active {
    transform: translateY(0);
}

/* ---- Metric cards ---- */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 1rem 1.1rem;
    box-shadow: 0 3px 12px rgba(17, 24, 39, 0.06);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(17, 24, 39, 0.10);
}
[data-testid="stMetricLabel"] p {
    font-weight: 600 !important;
    color: #4b5563 !important;
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
[data-testid="stMetricValue"] {
    font-weight: 800 !important;
    color: #111827 !important;
    font-size: 1.55rem !important;
}
[data-testid="stMetricDelta"] {
    font-weight: 600 !important;
}

/* ---- Result hero ---- */
.result-hero {
    font-size: 2rem;
    font-weight: 800;
    text-align: center;
    margin: 1.2rem 0 0.4rem 0;
    background: linear-gradient(135deg, #0369a1 0%, #4338ca 55%, #be185d 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
    line-height: 1.2;
}
.result-subhero {
    text-align: center;
    color: #4b5563;
    font-size: 0.95rem;
    margin-bottom: 1.2rem;
}

/* ---- Success banner ---- */
.plan-banner {
    background: #ffffff;
    color: #065f46;
    border: 1px solid #6ee7b7;
    border-left: 5px solid #10b981;
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    margin: 0.6rem 0 1.4rem 0;
    font-weight: 600;
    text-align: left;
    box-shadow: 0 3px 10px rgba(16, 185, 129, 0.12);
}

/* ---- Subsection titles ---- */
.subsection-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #111827;
    margin: 1.8rem 0 0.8rem 0;
    letter-spacing: -0.01em;
}

/* ---- Meal cards ---- */
.meal-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-left: 4px solid #7c3aed;
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    margin: 0.7rem 0;
    box-shadow: 0 4px 14px rgba(17, 24, 39, 0.07);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.meal-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(17, 24, 39, 0.10);
}
.meal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.4rem;
    flex-wrap: wrap;
    gap: 0.4rem;
}
.meal-slot {
    font-size: 0.78rem;
    font-weight: 700;
    color: #4338ca;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.meal-calories {
    font-size: 0.82rem;
    color: #4b5563;
    font-weight: 500;
}
.meal-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 0.55rem;
    line-height: 1.3;
}
.meal-list {
    color: #374151;
    margin: 0;
    padding-left: 1.2rem;
    font-size: 0.95rem;
    line-height: 1.7;
}

/* ---- Info / how-it-works card ---- */
.info-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-left: 4px solid #4338ca;
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    color: #1f2937;
    font-size: 0.95rem;
    line-height: 1.7;
    margin: 1rem 0;
    box-shadow: 0 3px 12px rgba(17, 24, 39, 0.06);
}

/* ---- Footer ---- */
.app-footer {
    text-align: center;
    color: #6b7280;
    font-size: 0.85rem;
    margin-top: 3rem;
    padding-top: 1.2rem;
    border-top: 1px solid #e5e7eb;
}

/* ---- Warning / alert boxes ---- */
.stAlert {
    border-radius: 12px !important;
}

/* ---- Radio / checkbox text ---- */
.stRadio [role="radiogroup"] label p,
.stCheckbox label p {
    color: #1f2937 !important;
    font-weight: 500 !important;
}

/* ---- Slider: color only, DO NOT touch layout / pointer-events ---- */
/* Track filled portion */
.stSlider [data-baseweb="slider"] > div > div > div > div {
    background: #7c3aed !important;
}
/* Thumb */
.stSlider [role="slider"] {
    background: #ffffff !important;
    border: 3px solid #7c3aed !important;
    box-shadow: 0 2px 6px rgba(124, 58, 237, 0.35) !important;
}
/* Value tooltip */
.stSlider [data-baseweb="tooltip"] {
    background: #4338ca !important;
    color: #ffffff !important;
}

/* ---- Spinner ---- */
.stSpinner > div {
    border-top-color: #7c3aed !important;
}
</style>
"""

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
MODEL_PATH = "calorie_model.pkl"

ACTIVITY_MULTIPLIERS = {
    "Sedentary (little or no exercise)": 1.2,
    "Lightly active (1-3 days per week)": 1.375,
    "Moderately active (3-5 days per week)": 1.55,
    "Very active (6-7 days per week)": 1.725,
}

GOALS = ["Weight Loss", "Weight Gain", "Muscle Building", "Maintain Weight"]

MEAL_SPLITS = {
    2: [("Breakfast", 0.45), ("Dinner", 0.55)],
    3: [("Breakfast", 0.30), ("Lunch", 0.40), ("Dinner", 0.30)],
    4: [("Breakfast", 0.25), ("Lunch", 0.30), ("Snack", 0.15), ("Dinner", 0.30)],
    5: [
        ("Breakfast", 0.22),
        ("Mid-morning Snack", 0.10),
        ("Lunch", 0.28),
        ("Afternoon Snack", 0.10),
        ("Dinner", 0.30),
    ],
}

MEAL_TAG_MAP = {
    "Breakfast": "breakfast",
    "Lunch": "lunch",
    "Dinner": "dinner",
    "Snack": "snack",
    "Mid-morning Snack": "snack",
    "Afternoon Snack": "snack",
}

MEAL_EMOJI = {
    "Breakfast": "🌅",
    "Lunch": "🥗",
    "Dinner": "🍽️",
    "Snack": "🥨",
    "Mid-morning Snack": "☕",
    "Afternoon Snack": "🍎",
}


# -----------------------------------------------------------------------------
# Machine Learning Pipeline (UNCHANGED BACKEND)
# -----------------------------------------------------------------------------
def _mifflin_st_jeor(age: int, gender: str, height: float, weight: float) -> float:
    """Calculate Basal Metabolic Rate using Mifflin-St Jeor equation."""
    if gender == "Male":
        return 10 * weight + 6.25 * height - 5 * age + 5
    return 10 * weight + 6.25 * height - 5 * age - 161


@st.cache_resource(show_spinner=False)
def train_model():
    """
    Load a pre-trained model from disk if it exists; otherwise, synthesize a
    dataset of >= 1000 rows from the Mifflin-St Jeor BMR formula with activity
    multipliers and random noise, one-hot encode categorical features, train a
    LinearRegression model, and persist it with joblib.
    """
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass

    rng = np.random.default_rng(42)
    n_rows = 1200

    ages = rng.integers(15, 71, size=n_rows)
    heights = rng.integers(140, 211, size=n_rows)
    weights = rng.integers(40, 151, size=n_rows)
    genders = rng.choice(["Male", "Female"], size=n_rows)
    activities = rng.choice(list(ACTIVITY_MULTIPLIERS.keys()), size=n_rows)
    goals = rng.choice(GOALS, size=n_rows)

    calories = np.empty(n_rows, dtype=float)
    for i in range(n_rows):
        bmr = _mifflin_st_jeor(
            int(ages[i]), str(genders[i]), float(heights[i]), float(weights[i])
        )
        tdee = bmr * ACTIVITY_MULTIPLIERS[str(activities[i])]
        calories[i] = tdee + rng.normal(0.0, 60.0)

    df = pd.DataFrame(
        {
            "age": ages,
            "gender": genders,
            "height": heights,
            "weight": weights,
            "activity": activities,
            "goal": goals,
            "calories": calories,
        }
    )

    X = pd.get_dummies(
        df.drop(columns=["calories"]),
        columns=["gender", "activity", "goal"],
    )
    y = df["calories"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    bundle = {
        "model": model,
        "columns": X.columns.tolist(),
        "train_score": float(model.score(X_train, y_train)),
        "test_score": float(model.score(X_test, y_test)),
    }

    try:
        joblib.dump(bundle, MODEL_PATH)
    except Exception:
        pass

    return bundle


def predict_calories(
    bundle: dict,
    age: int,
    gender: str,
    height: float,
    weight: float,
    activity: str,
    goal: str,
) -> float:
    """Predict daily maintenance calories for a single user."""
    user_df = pd.DataFrame(
        [
            {
                "age": age,
                "gender": gender,
                "height": height,
                "weight": weight,
                "activity": activity,
                "goal": goal,
            }
        ]
    )
    X = pd.get_dummies(user_df, columns=["gender", "activity", "goal"])
    X = X.reindex(columns=bundle["columns"], fill_value=0)
    return float(bundle["model"].predict(X)[0])


# -----------------------------------------------------------------------------
# Macro calculation (UNCHANGED BACKEND)
# -----------------------------------------------------------------------------
def calculate_macros(calories: int, goal: str):
    """Return (protein_g, carbs_g, fats_g) rounded to whole grams."""
    if goal == "Weight Loss":
        p_pct, c_pct, f_pct = 0.35, 0.35, 0.30
    elif goal in ("Muscle Building", "Weight Gain"):
        p_pct, c_pct, f_pct = 0.30, 0.45, 0.25
    else:
        p_pct, c_pct, f_pct = 0.30, 0.40, 0.30

    protein_g = round((calories * p_pct) / 4)
    carbs_g = round((calories * c_pct) / 4)
    fats_g = round((calories * f_pct) / 9)
    return protein_g, carbs_g, fats_g


# -----------------------------------------------------------------------------
# Food database & meal planning
# -----------------------------------------------------------------------------
def _build_food_dataset() -> pd.DataFrame:
    """
    Clean, generic, globally common fitness-friendly food database.
    No beef, pork, or red meat of any kind is included.
    No region-specific Indian dishes are included (Paneer retained).
    Each row: name, calories per serving, category (veg|egg|nonveg), meal_tags.
    """
    foods = [
        # --- Breakfast staples ---
        ("Rolled oats", 150, "veg", ["breakfast"]),
        ("Overnight oats", 210, "veg", ["breakfast"]),
        ("Granola", 180, "veg", ["breakfast", "snack"]),
        ("Greek yogurt", 120, "veg", ["breakfast", "snack"]),
        ("Skim milk", 90, "veg", ["breakfast"]),
        ("Almond milk", 40, "veg", ["breakfast"]),
        ("Peanut butter", 190, "veg", ["breakfast", "snack"]),
        ("Whole wheat bread", 140, "veg", ["breakfast", "lunch"]),
        ("Sourdough toast", 160, "veg", ["breakfast", "lunch"]),
        ("Avocado toast", 240, "veg", ["breakfast", "lunch"]),
        ("Eggs", 150, "egg", ["breakfast", "lunch"]),
        ("Egg whites", 80, "egg", ["breakfast"]),
        ("Scrambled eggs", 180, "egg", ["breakfast"]),
        ("Cottage cheese", 120, "veg", ["breakfast", "snack"]),
        ("Protein pancakes", 260, "egg", ["breakfast"]),
        ("Protein yogurt parfait", 230, "veg", ["breakfast", "snack"]),

        # --- Plant proteins (global) ---
        ("Tofu", 160, "veg", ["lunch", "dinner"]),
        ("Tofu stir fry", 260, "veg", ["lunch", "dinner"]),
        ("Tempeh", 190, "veg", ["lunch", "dinner"]),
        ("Paneer", 220, "veg", ["breakfast", "lunch", "dinner"]),
        ("Seitan", 170, "veg", ["lunch", "dinner"]),
        ("Chickpeas", 180, "veg", ["lunch", "dinner", "snack"]),
        ("Lentils", 180, "veg", ["lunch", "dinner"]),
        ("Black beans", 180, "veg", ["lunch", "dinner"]),
        ("Kidney beans", 175, "veg", ["lunch", "dinner"]),
        ("Hummus", 150, "veg", ["lunch", "snack"]),
        ("Falafel", 200, "veg", ["lunch", "dinner"]),
        ("Edamame", 120, "veg", ["lunch", "snack"]),

        # --- Complex carbs ---
        ("Quinoa", 220, "veg", ["lunch", "dinner"]),
        ("Quinoa bowl", 330, "veg", ["lunch", "dinner"]),
        ("Brown rice", 215, "veg", ["lunch", "dinner"]),
        ("Whole grain pasta", 220, "veg", ["lunch", "dinner"]),
        ("Sweet potato", 180, "veg", ["lunch", "dinner"]),
        ("Couscous", 175, "veg", ["lunch", "dinner"]),
        ("Farro", 180, "veg", ["lunch", "dinner"]),
        ("Bulgur wheat", 160, "veg", ["lunch", "dinner"]),
        ("Cauliflower rice", 40, "veg", ["lunch", "dinner"]),
        ("Zucchini noodles", 30, "veg", ["lunch", "dinner"]),

        # --- Vegetables & salads ---
        ("Avocado", 160, "veg", ["breakfast", "lunch", "snack"]),
        ("Broccoli", 55, "veg", ["lunch", "dinner"]),
        ("Spinach", 30, "veg", ["lunch", "dinner"]),
        ("Kale", 40, "veg", ["lunch", "dinner"]),
        ("Bell peppers", 45, "veg", ["lunch", "dinner"]),
        ("Zucchini", 35, "veg", ["lunch", "dinner"]),
        ("Mushrooms", 40, "veg", ["lunch", "dinner"]),
        ("Mixed greens", 30, "veg", ["lunch", "dinner"]),
        ("Garden salad", 120, "veg", ["lunch", "dinner"]),
        ("Caesar salad", 180, "veg", ["lunch", "dinner"]),

        # --- Animal proteins (NO red meat — poultry & seafood only) ---
        ("Grilled chicken breast", 230, "nonveg", ["lunch", "dinner"]),
        ("Grilled chicken salad", 300, "nonveg", ["lunch", "dinner"]),
        ("Turkey breast", 190, "nonveg", ["lunch", "dinner"]),
        ("Turkey sandwich", 320, "nonveg", ["lunch"]),
        ("Tuna", 180, "nonveg", ["lunch", "dinner"]),
        ("Tuna bowl", 320, "nonveg", ["lunch", "dinner"]),
        ("Salmon", 280, "nonveg", ["lunch", "dinner"]),
        ("Grilled salmon with vegetables", 360, "nonveg", ["lunch", "dinner"]),
        ("Shrimp", 150, "nonveg", ["lunch", "dinner"]),
        ("White fish", 180, "nonveg", ["lunch", "dinner"]),
        ("Cod", 170, "nonveg", ["lunch", "dinner"]),

        # --- Supplements / shakes ---
        ("Whey protein powder (1 scoop = 25 g protein)", 120, "veg",
         ["breakfast", "snack"]),
        ("Whey protein shake", 150, "veg", ["breakfast", "snack"]),
        ("Fruit smoothie", 220, "veg", ["breakfast", "snack"]),
        ("Protein smoothie", 260, "veg", ["breakfast", "snack"]),
        ("Protein bar", 210, "veg", ["snack"]),

        # --- Fruits & snacks ---
        ("Fruit bowl", 180, "veg", ["breakfast", "snack"]),
        ("Berries", 70, "veg", ["breakfast", "snack"]),
        ("Apple", 95, "veg", ["snack", "breakfast"]),
        ("Banana", 105, "veg", ["breakfast", "snack"]),
        ("Orange", 85, "veg", ["breakfast", "snack"]),
        ("Trail mix", 220, "veg", ["snack"]),
        ("Dark chocolate", 170, "veg", ["snack"]),
        ("Walnuts", 190, "veg", ["snack"]),
        ("Almonds", 170, "veg", ["snack"]),
        ("Roasted chickpeas", 160, "veg", ["snack"]),
        ("Rice cakes", 70, "veg", ["snack"]),
    ]
    return pd.DataFrame(foods, columns=["name", "calories", "category", "meal_tags"])


def _filter_foods(df: pd.DataFrame, diet_type: str, include_eggs: bool) -> pd.DataFrame:
    if diet_type == "Non Vegetarian":
        return df.copy()
    if include_eggs:
        return df[df["category"].isin(["veg", "egg"])].copy()
    return df[df["category"] == "veg"].copy()


def _pretty_meal_title(items: list, slot: str) -> str:
    """Create cohesive, restaurant-style meal names from the selected items."""
    if not items:
        return slot
    if len(items) == 1:
        return items[0]

    lower_items = [it.lower() for it in items]

    presets = [
        ({"greek yogurt", "granola", "berries"}, "Greek yogurt with granola and berries"),
        ({"sourdough toast", "avocado", "eggs"}, "Avocado toast with eggs"),
        ({"whole wheat bread", "avocado", "eggs"}, "Avocado toast with eggs"),
        ({"grilled chicken breast", "quinoa"}, "Grilled chicken quinoa bowl"),
        ({"tofu", "bell peppers"}, "Tofu vegetable stir fry"),
        ({"turkey breast", "mixed greens"}, "Turkey sandwich with salad"),
        ({"salmon", "broccoli"}, "Salmon with roasted vegetables"),
        ({"fruit smoothie", "banana", "peanut butter"},
         "Protein smoothie with banana and peanut butter"),
        ({"whey protein shake", "banana"}, "Banana protein shake"),
        ({"rolled oats", "peanut butter", "banana"}, "Peanut butter banana oatmeal"),
        ({"brown rice", "black beans"}, "Rice & beans bowl"),
    ]

    item_set = set(lower_items)
    for required, title in presets:
        if required.issubset(item_set):
            return title

    if len(items) == 2:
        return f"{items[0]} with {items[1].lower()}"
    return f"{items[0]} with {', '.join(x.lower() for x in items[1:-1])} and {items[-1].lower()}"


def _pick_meal(pool: pd.DataFrame, target_cal: float, tag: str,
               used: set, max_attempts: int = 250):
    """Sample foods tagged for this meal until total calories are within ±10%."""
    candidates = pool[pool["meal_tags"].apply(lambda t: tag in t)]
    if candidates.empty:
        candidates = pool

    fresh = candidates[~candidates["name"].isin(used)]
    if len(fresh) >= 3:
        candidates = fresh

    best_pick = None
    best_diff = float("inf")
    low, high = target_cal * 0.9, target_cal * 1.1

    for _ in range(max_attempts):
        n_items = random.choice([2, 3, 3, 4])
        n_items = min(n_items, len(candidates))
        if n_items == 0:
            break
        picks = candidates.sample(n=n_items)
        total = picks["calories"].sum()
        diff = abs(total - target_cal)
        if diff < best_diff:
            best_diff = diff
            best_pick = picks
        if low <= total <= high:
            return picks, int(total)

    if best_pick is None:
        return None, 0
    return best_pick, int(best_pick["calories"].sum())


def generate_meal_plan(calories: int, meals_per_day: int,
                       diet_type: str, include_eggs: bool):
    """Return a list of meal dicts for the daily plan."""
    food_df = _build_food_dataset()
    pool = _filter_foods(food_df, diet_type, include_eggs)

    splits = MEAL_SPLITS[meals_per_day]
    used_names: set = set()
    plan = []

    for slot_name, pct in splits:
        target = calories * pct
        tag = MEAL_TAG_MAP[slot_name]
        picks, total = _pick_meal(pool, target, tag, used_names)
        items = picks["name"].tolist() if picks is not None else []
        for item in items:
            used_names.add(item)
        plan.append(
            {
                "slot": slot_name,
                "target": int(round(target)),
                "actual": int(total),
                "items": items,
                "title": _pretty_meal_title(items, slot_name),
            }
        )
    return plan


# -----------------------------------------------------------------------------
# Workout recommendation (UNCHANGED BACKEND)
# -----------------------------------------------------------------------------
def _determine_workouts(cal_adjustment: float):
    abs_adj = abs(cal_adjustment)
    if abs_adj < 300:
        return 3, "Full-body training (Mon / Wed / Fri)"
    if abs_adj < 600:
        return 4, "Upper / Lower split (Mon / Tue / Thu / Fri)"
    return 6, "Push / Pull / Legs × 2 — 6 sessions per week"


# -----------------------------------------------------------------------------
# UI Helpers
# -----------------------------------------------------------------------------
def render_meal_card(meal: dict):
    """Render a single meal as a styled HTML card."""
    emoji = MEAL_EMOJI.get(meal["slot"], "🍴")
    items_html = "".join(f"<li>{item}</li>" for item in meal["items"])
    html = f"""
    <div class="meal-card">
        <div class="meal-header">
            <span class="meal-slot">{emoji}&nbsp;&nbsp;{meal['slot']}</span>
            <span class="meal-calories">{meal['actual']} kcal · target {meal['target']} kcal</span>
        </div>
        <div class="meal-title">{meal['title']}</div>
        <ul class="meal-list">{items_html}</ul>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Main Streamlit UI
# -----------------------------------------------------------------------------
def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ----- HERO -----
    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero-badge">🤖 AI · Machine Learning · Nutrition Science</div>
            <div class="hero-title">AI Personalized Diet &amp; Fitness Planner</div>
            <div class="hero-subtitle">
                Enter your details to receive a scientifically calculated diet and
                exercise recommendation powered by AI.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ----- Name (top of page) -----
    with st.container(border=True):
        st.markdown(
            '<div class="section-title">👋 Welcome</div>',
            unsafe_allow_html=True,
        )
        user_name = st.text_input(
            "Your Name",
            placeholder="Enter your name",
            label_visibility="collapsed",
        )

    # Warm up the model on first run (cached)
    bundle = train_model()

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------------
    # Section 1 — User Body Information
    # --------------------------------------------------------------
    with st.container(border=True):
        st.markdown(
            '<div class="section-title">🧍 User Body Information</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-caption">Tell us a little about yourself so the model can personalize your metabolic profile.</div>',
            unsafe_allow_html=True,
        )
        left, right = st.columns(2, gap="large")
        with left:
            age = st.number_input("Age", min_value=15, max_value=70, value=22, step=1)
            height = st.number_input(
                "Height (cm)", min_value=140, max_value=210, value=170, step=1
            )
            weight = st.number_input(
                "Weight (kg)", min_value=40, max_value=150, value=70, step=1
            )
        with right:
            gender = st.selectbox("Gender", ["Male", "Female"])
            activity = st.selectbox(
                "Activity Level",
                list(ACTIVITY_MULTIPLIERS.keys()),
                index=2,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------------
    # Section 2 — Goals and Preferences
    # --------------------------------------------------------------
    with st.container(border=True):
        st.markdown(
            '<div class="section-title">🎯 Goals &amp; Preferences</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-caption">Choose your goal and dietary preferences — we will tailor the plan to match.</div>',
            unsafe_allow_html=True,
        )

        goal = st.selectbox("Primary Goal", GOALS, index=0)

        # Target weight change & timeframe are only relevant for non-maintenance
        # goals. Hide them entirely when "Maintain Weight" is selected and
        # default to safe sentinel values for the backend.
        if goal != "Maintain Weight":
            c1, c2 = st.columns(2, gap="large")
            with c1:
                target_change = st.number_input(
                    "Target Weight Change (kg)",
                    min_value=0.0, max_value=50.0, value=5.0, step=0.5,
                )
            with c2:
                months = st.number_input(
                    "Time to Achieve Goal (months)",
                    min_value=0, max_value=36, value=3, step=1,
                )
        else:
            target_change = 0.0
            months = 1  # safe placeholder; never used when goal is Maintain

        diet_type = st.radio(
            "Diet Type", ["Vegetarian", "Non Vegetarian"], horizontal=True
        )
        include_eggs = False
        if diet_type == "Vegetarian":
            include_eggs = st.checkbox("Include Eggs", value=True)

        meals_per_day = st.selectbox(
            "Meals Per Day",
            options=[2, 3, 4, 5],
            index=2,  # default = 4
            key="meals_per_day_select",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------------
    # Section 3 — Generate Plan and Results
    # --------------------------------------------------------------
    with st.container(border=True):
        st.markdown(
            '<div class="section-title">✨ Generate Plan &amp; Results</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-caption">Click below and our AI will build a complete daily nutrition and training plan in seconds.</div>',
            unsafe_allow_html=True,
        )
        clicked = st.button(
            "🚀  Generate My Plan",
            type="primary",
            use_container_width=True,
        )

    # --------------------------------------------------------------
    # Results
    # --------------------------------------------------------------
    if clicked:
        # ---- Input validation (only relevant for non-maintenance goals) ----
        if goal != "Maintain Weight":
            if months <= 0:
                st.warning(
                    "⚠️ Time to achieve your goal must be at least 1 month. "
                    "Please adjust your inputs."
                )
                return
            if target_change <= 0:
                st.warning(
                    "⚠️ Target weight change must be greater than zero for "
                    "your selected goal."
                )
                return
            weekly_rate = (target_change / months) / 4.33
            if weekly_rate > 1.0:
                st.warning(
                    "⚠️ That weight change rate exceeds 1 kg per week, which "
                    "is considered unsafe. Please extend your timeframe or "
                    "lower your target for sustainable, healthy results."
                )
                return

        with st.spinner(
            "Training AI model and generating your personalized plan..."
        ):
            maintenance = predict_calories(
                bundle, age, gender, height, weight, activity, goal
            )

            # Safely compute the daily caloric adjustment. For Maintain Weight,
            # the adjustment is zero and target == maintenance.
            if goal == "Maintain Weight":
                daily_adj = 0.0
                target_cal = maintenance
                direction = 0.0
            else:
                total_cal_change = 7700.0 * float(target_change)
                daily_adj = (total_cal_change / float(months)) / 30.0
                if goal == "Weight Loss":
                    target_cal = maintenance - daily_adj
                    direction = -daily_adj
                else:  # Weight Gain or Muscle Building
                    target_cal = maintenance + daily_adj
                    direction = daily_adj

            target_cal = int(round(float(np.clip(target_cal, 1200, 4000))))
            maintenance = int(round(maintenance))

            protein_g, carbs_g, fats_g = calculate_macros(target_cal, goal)
            workouts, split_plan = _determine_workouts(direction)
            meal_plan = generate_meal_plan(
                target_cal, meals_per_day, diet_type, include_eggs
            )

        # ---- Personalized header ----
        name_clean = user_name.strip() if user_name else ""
        if name_clean:
            header = f"{name_clean}'s Personalized Diet &amp; Fitness Plan"
        else:
            header = "Your Personalized Diet &amp; Fitness Plan"

        st.markdown(
            f'<div class="result-hero">{header}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="result-subhero">A science-backed daily plan tailored to your goals</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="plan-banner">✅ Your personalized plan is ready — scroll down to see your daily targets and meals.</div>',
            unsafe_allow_html=True,
        )

        # ---- Metric row ----
        st.markdown(
            '<div class="subsection-title">📊 Daily Targets</div>',
            unsafe_allow_html=True,
        )

        row1 = st.columns(3, gap="medium")
        row1[0].metric("Maintenance Calories", f"{maintenance:,} kcal")
        delta = (
            f"{int(target_cal - maintenance):+,} kcal"
            if target_cal != maintenance else None
        )
        row1[1].metric("Target Calories", f"{target_cal:,} kcal", delta=delta)
        row1[2].metric("Workouts / Week", f"{workouts}")

        row2 = st.columns(3, gap="medium")
        row2[0].metric("Protein", f"{protein_g} g")
        row2[1].metric("Carbs", f"{carbs_g} g")
        row2[2].metric("Fats", f"{fats_g} g")

        # ---- Workout split ----
        st.markdown("<br>", unsafe_allow_html=True)
        st.success(f"🏋️  **Recommended Training Split:** {split_plan}")

        # ---- Meal plan ----
        st.markdown(
            '<div class="subsection-title">🍽️ Your Daily Meal Plan</div>',
            unsafe_allow_html=True,
        )
        for meal in meal_plan:
            render_meal_card(meal)

        # ---- How it works ----
        st.markdown(
            '<div class="subsection-title">🧠 How this plan was created</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="info-card">
            This plan combines <b>machine learning</b>, <b>metabolic science</b>,
            and a <b>recommendation engine</b>. A Linear Regression model was
            trained on a synthetic dataset of 1,000+ samples built from the
            <b>Mifflin–St Jeor BMR equation</b> and standard activity multipliers
            (1.2, 1.375, 1.55, 1.725) to predict your daily maintenance calories.
            Those are then adjusted using the <b>7,700 kcal per kilogram</b> rule
            scaled to your goal and timeframe. Macronutrient ratios are selected
            based on your objective (weight loss, muscle gain, or maintenance)
            and converted to grams. The meal recommender then samples from a
            curated database of global fitness-friendly foods, filtered to your
            diet preferences, to construct meals whose calorie totals stay
            within <b>±10%</b> of each meal-slot target.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---- Footer ----
    st.markdown(
        '<div class="app-footer">Built using Streamlit, Machine Learning and Nutrition Science</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
