
# 🩺 MedSense — AI-Powered Medical Assistant Application

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

## 📦 Run Locally

Follow these steps to run the project on your local machine:

### 1. Clone the Repository
```bash
git clone https://github.com/WaseemKhan09/medisense.git
cd medisense
```

### 2. Create & Activate Virtual Environment
```bash
# On Linux/Mac:
python3 -m venv venv
source venv/bin/activate

# On Windows:
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables
Create a `.env` file in the root directory with the following:
```env
HF_TOKEN=your_huggingface_api_key_here
```

### 5. Run the Flask Web App
```bash
flask run
```

### 6. Run the Chatbot (Streamlit Interface)
In a new terminal:
```bash
streamlit run medibot.py
```

The Flask app will run your core project, and the Streamlit app will launch the chatbot interface in a browser.

---

## 🙋‍♀️ Who Can Use This?

- Medical students or enthusiasts learning about AI in healthcare  
- Developers exploring LangChain + Hugging Face integrations  
- Institutions needing a basic AI-powered symptom-checker prototype  
- Anyone interested in ML/NLP-based chatbot systems  

---

## 🙏 Acknowledgements

Thanks to:

- [LangChain](https://www.langchain.com/)
- [Hugging Face](https://huggingface.co/)
- [Streamlit](https://streamlit.io/)
- [scikit-learn](https://scikit-learn.org/)

---

## 📧 Contact

For queries, feedback, or collaboration:  
📩 **waseem7861khan@gmail.com**

---

⭐ If you found this helpful, please consider giving it a star!


