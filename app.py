import os
from flask import Flask, render_template, jsonify, request, send_from_directory
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.llms import Replicate
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from werkzeug.utils import secure_filename
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load environment variables
load_dotenv()

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set up the language model
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    raise ValueError("No REPLICATE_API_TOKEN found. Please set it in your .env file.")

os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

llm = Replicate(
    model="a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea",
    config={
        'max_new_tokens': 100,
        'temperature': 0.7,
        'top_k': 50
    }
)

# PDF processing functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_pdf(file_path):
    all_text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            for page in reader.pages:
                all_text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error reading PDF: {str(e)}")
        return None
    return all_text if all_text else None

def text_split(text):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    document = Document(page_content=text)
    return text_splitter.split_documents([document])

# Flask routes
@app.route("/")
def index():
    return render_template('health.html')

@app.route("/info", methods=["POST"])
def get_info():
    try:
        msg = request.form.get("msg", "")
        input_text = msg
        print(f"Received message: {input_text}")

        # Handle PDF upload
        pdf_content = None
        if 'pdf' in request.files:
            file = request.files['pdf']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                # Process the uploaded PDF
                pdf_content = load_pdf(file_path)
                if pdf_content:
                    text_chunks = text_split(pdf_content)
                    print(f"PDF processed: {len(text_chunks)} chunks")
                else:
                    print("Failed to process PDF")

        # Combine user input with PDF content if available
        if pdf_content:
            input_text += f"\n\nAdditional context from uploaded PDF:\n{pdf_content[:500]}..."

        # Retrieve information from the model
        print("Generating response from LLM...")
        result = llm.generate([input_text])
        print(f"LLMResult: {result}")

        # Access the generated text from the result object
        if result.generations and result.generations[0]:
            generated_text = result.generations[0][0].text
        else:
            generated_text = "No information available."

        print(f"Response: {generated_text}")

        return generated_text
    except Exception as e:
        print(f"Error in get_info: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)