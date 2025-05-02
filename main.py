from flask import Flask, request, render_template, jsonify, url_for, flash, redirect, session
from flask_mail import Mail, Message
import numpy as np
import pandas as pd
import pickle
import ast
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user


app = Flask(__name__)
app.secret_key = '6a15fcbc2e19eeea98c60e994489e8cb'

# Database setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

# Login Manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # If user not logged in, redirect to login page

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# Create database
with app.app_context():
    db.create_all()

# Load user
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Flask-Mail Configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'waseem7861khan@gmail.com'
app.config['MAIL_PASSWORD'] = 'mdkr tgiy hsag mkdm'

mail = Mail(app)

# Load model
rf = pickle.load(open('models/random_forest.pkl', 'rb'))

# ============================ Helper Function ============================

def get_datasets():
    patient_type = session.get('patient_type', 'normal')

    precautions = pd.read_csv(f"datasets/{patient_type}/precautions_df.csv")
    workout = pd.read_csv(f"datasets/{patient_type}/workout_df.csv")
    medications = pd.read_csv(f"datasets/{patient_type}/medications.csv")
    diets = pd.read_csv(f"datasets/{patient_type}/diets.csv")
    description = pd.read_csv(f"datasets/{patient_type}/description.csv", encoding='latin-1')  # Common for all

    return precautions, workout, medications, diets, description

def helper(dis):
    precautions, workout, medications, diets, description = get_datasets()

    desc_row = description[description['Disease'] == dis]['Description']
    desc = " ".join(desc_row.values) if not desc_row.empty else "Description not available"

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values] if not pre.empty else [["No Precautions Found"]]

    med_row = medications[medications['Disease'] == dis]['Medication']
    med = ast.literal_eval(med_row.values[0]) if not med_row.empty else []

    diet_row = diets[diets['Disease'] == dis]['Diet']
    die = ast.literal_eval(diet_row.values[0]) if not diet_row.empty else []

    wrkout_row = workout[workout['disease'] == dis]['workout']
    wrkout = wrkout_row.values if not wrkout_row.empty else ["No Workout Recommendation"]

    return desc, pre, med, die, wrkout

symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {0: '(vertigo) Paroymsal  Positional Vertigo', 1: 'AIDS', 2: 'Acne', 3: 'Alcoholic hepatitis', 4: 'Allergy', 5: 'Arthritis', 6: 'Bronchial Asthma', 7: 'Cervical spondylosis', 8: 'Chicken pox', 9: 'Chronic cholestasis', 10: 'Common Cold', 11: 'Dengue', 12: 'Diabetes ', 13: 'Dimorphic hemmorhoids(piles)', 14: 'Drug Reaction', 15: 'Fungal infection', 16: 'GERD', 17: 'Gastroenteritis', 18: 'Heart attack', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 23: 'Hypertension ', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 26: 'Hypothyroidism', 27: 'Impetigo', 28: 'Jaundice', 29: 'Malaria', 30: 'Migraine', 31: 'Osteoarthristis', 32: 'Paralysis (brain hemorrhage)', 33: 'Peptic ulcer diseae', 34: 'Pneumonia', 35: 'Psoriasis', 36: 'Tuberculosis', 37: 'Typhoid', 38: 'Urinary tract infection', 39: 'Varicose veins', 40: 'hepatitis A'}


def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[rf.predict([input_vector])[0]]

# ============================ Routes ============================


@app.route("/")
@login_required
def landing_page():
    return redirect(url_for('select_patient'))  # <-- First thing user sees


@app.route('/select_patient', methods=['GET', 'POST'])
def select_patient():
    if request.method == 'POST':
        selected_type = request.form.get('patient_type')
    else:
        selected_type = request.args.get('ptype')

    if not selected_type:
        return render_template('select_patient.html')   # render selection page instead of redirect

    if selected_type not in ['normal', 'diabetic', 'bp']:
        flash("Invalid patient type selected!", "error")
        return render_template('select_patient.html')

    session['patient_type'] = selected_type
    print("Selected patient type:", session['patient_type'])
    return redirect(url_for('index'))


@app.route("/index")
@login_required
def index():
    symptoms = list(symptoms_dict.keys())
    return render_template("index.html", symptoms=symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    if 'patient_type' not in session:
        return redirect(url_for('select_patient'))

    user_symptoms = request.form.getlist('symptoms')
    if not user_symptoms:
        message = "Please select at least one symptom."
        symptoms = list(symptoms_dict.keys())
        return render_template('index.html', message=message, symptoms=symptoms)

    predicted_disease = get_predicted_value(user_symptoms)
    dis_des, pre, med, die, wrkout = helper(predicted_disease)

    my_precautions = [i for i in pre[0]]

    return render_template('index.html',
                           predicted_disease=predicted_disease,
                           dis_des=dis_des,
                           my_precautions=my_precautions,
                           medications=med,
                           my_diet=die,
                           workout=wrkout,
                           symptoms=list(symptoms_dict.keys())
                           )


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/contact')
def contact():
    return render_template("contact.html")


# Signup route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return "Username already exists! Try another."

        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for('login'))
    return render_template('signup.html')


# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username, password=password).first()
        if user:
            login_user(user)
            return redirect(url_for('select_patient'))
        else:
            return "Invalid credentials! Try again."

    return render_template('login.html')


# Logout route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/send_message', methods=['POST'])
def send_message():
    name = request.form['name']
    email = request.form['email']
    subject = request.form['subject']
    message_body = request.form['message']

    msg = Message(subject=f"Contact Form: {subject}",
                  sender=app.config['MAIL_USERNAME'],
                  recipients=['your_email@gmail.com'])
    msg.body = f"Name: {name}\nEmail: {email}\n\nMessage:\n{message_body}"

    try:
        mail.send(msg)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
