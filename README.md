# AI-based-Diet-and-Fitness-Planner
AI Personalized Diet & Fitness Planner
A single-page Machine Learning web application that generates a personalized diet plan, calorie target, macros, and workout recommendation based on user body details and fitness goals.
This project was built as a second-year AI/ML mini project using Python and Streamlit.
Project Overview
This application collects basic user information such as age, height, weight, activity level and goal, then predicts daily calorie needs using a Machine Learning model. Based on the predicted calories, the app generates:
• Target daily calories
• Protein, carbs and fat breakdown
• Workout frequency recommendation
• Full daily meal plan
The app runs locally and does not require login or database setup.
Features
• Clean single-page web interface
• Personalized calorie prediction using ML
• Automatic macro nutrient calculation
• Smart meal plan generator
• Workout frequency suggestion
• Runs fully offline on local machine
How It Works
User enters body details and goal
A trained Linear Regression model predicts maintenance calories
Calories are adjusted based on weight goal
Macros are calculated based on fitness goal
Meal plan is generated from a fitness food dataset
Final personalized plan is displayed instantly
Tech Stack
• Python
• Streamlit
• Pandas
• NumPy
• Scikit-learn
• Joblib
All tools used in this project are Python libraries.
Machine Learning Model
The app uses a Linear Regression model trained on ~1200 synthetic user records generated using:
• Mifflin–St Jeor BMR formula
• Activity multipliers
• Randomized realistic variations
The trained model is saved and reused to avoid retraining every time the app runs.
Installation & Setup

Clone the repository:
git clone <your-repo-link>
cd <repo-folder>
Create virtual environment:
python -m venv venv

Activate virtual environment:
Windows:
venv\Scripts\activate
Mac/Linux:
source venv/bin/activate
Install dependencies:
pip install -r requirements.txt
Run the app:
streamlit run app.py
Project Structure
app.py
calorie_model.pkl
requirements.txt
README.md
Educational Purpose
This project is built for learning Machine Learning pipeline, Streamlit web apps, and AI-based recommendation systems.
Future Improvements
• Use real nutrition dataset
• Add progress tracking
• Add user accounts and database
• Deploy to cloud
