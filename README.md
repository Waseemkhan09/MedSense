# MedSense
# 🩺  MedSense: A symptoms based disease prediction and recommendation model 

MedSense is an intelligent and interactive medical chatbot web application that helps users identify possible diseases based on their symptoms and provides tailored suggestions including precautions, medications, diets, and workouts.

It is built using **Flask**, **Streamlit**, **LangChain**, and **Hugging Face**, with integrated **patient-type handling**, **chat interface**, and **disease prediction** powered by machine learning. 💡

---

## 🚀 Features

✅ Disease prediction based on symptoms  
✅ Chatbot UI with natural interaction flow  
✅ Personalized suggestions (based on patient type: Normal / Diabetic / BP)  
✅ Precautions, Medications, Diet & Workout recommendations  
✅ User Sign Up & Login Authentication  
✅ Stylish and responsive UI with Bootstrap & Streamlit  
✅ Trained ML model and symptom-based prediction flow  
✅ Ready-to-use vector database with LangChain + HuggingFace

---

## 🔄 Project Workflow

1. 👤 **User selects patient type**: (Normal / Diabetic / BP)
2. 🧾 **User enters symptoms** via dropdowns (dynamic based on count)
3. 🤖 **ML model predicts possible disease**
4. 🧠 **Model** provides:
   - Precautions  
   - Medications  
   - Diet  
   - Workouts  
5. 💬 User can continue interacting via chat
6. 🔐 **Sign In / Sign Up** page added for user access control

---

## 🛠️ Technologies Used

- **Frontend**: HTML, CSS, Bootstrap, Streamlit  
- **Backend**: Flask  
- **ML & NLP**: scikit-learn, Pandas, LangChain, Hugging Face  
- **Database**: FAISS Vector Store  
- **Authentication**: Flask-Login  
- **UI Components**: Jinja2 templates, Streamlit custom components

---

## 📦 How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/medisense.git
   cd medisense
2.Create virtual environment & activate
  python -m venv venv
  source venv/bin/activate   # On Windows: venv\Scripts\activate
3.Install dependencies
  pip install -r requirements.txt
4.Set your environment variables
  Create a .env file in the root and add:
  HF_TOKEN=your_huggingface_api_token_here
5.Run the Flask App
   flask run
   streamlit run medibot.py

