Step 1 - open the folder in terminal 

Step 2 — create virtual environment
python -m venv venv

Activate it:
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac / Linux

Step 3 — install packages
pip install -r requirements.txt

Step 4 — add your Groq API key to .env
.env file should contain:  GROQ_API_KEY=gsk_xxxxxxxxxxxx

Step 5 — run!
streamlit run app.py
