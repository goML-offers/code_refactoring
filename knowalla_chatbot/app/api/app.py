from fastapi import FastAPI, File, UploadFile
from transformers import T5ForConditionalGeneration, T5Tokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import boto3
import botocore
import os
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile
import uvicorn
from langchain.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer, util
from typing import List
import pinecone

# Load environment variables from .env file
load_dotenv()

AWS_ACCESS_KEY_ID = os.environ.get('S3_ACCESS_KEY')
AWS_SECRET_ACCESS_KEY = os.environ.get('S3_SECRET_KEY')
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')  # Your Pinecone API key
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')  # Your Pinecone environment name

# Initialize a connection to your S3 bucket
s3 = boto3.client("s3", aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
s3_bucket_name = S3_BUCKET_NAME

# Load the pre-trained T5 question generation model and tokenizer
question_model = T5ForConditionalGeneration.from_pretrained("allenai/t5-small-squad2-question-generation")
question_tokenizer = T5Tokenizer.from_pretrained("allenai/t5-small-squad2-question-generation")

# Initialize a text splitter
tokenizer = T5Tokenizer.from_pretrained("t5-small")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=24,
    length_function=lambda text: len(tokenizer.encode(text))
)

# Load a Sentence Transformer model
sentence_transformer_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Define the minimum matching cosine similarity for a correct answer
min_matching_similarity = 0.3  # Adjust as needed

# Initialize Pinecone client
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

# Define the Pinecone index name
index_name = "pinecone"
print("index_name is:")
print(index_name)

# Create a class to manage the state of the chatbot
class QAChatbot:
    def __init__(self):
        self.chunks = []  # Store document chunks
        self.question_index = 0  # Index for the current chunk
        self.questions = []  # Store generated questions
        self.answers = []  # Store user answers

    def generate_question(self, chunk_text):
        # Generate a question from the current chunk
        input_text = f"Generate a question from the following text: {chunk_text}"
        generated_question = question_model.generate(
            question_tokenizer.encode(input_text, return_tensors="pt"),
            max_length=128,  # Increase max_length
            num_return_sequences=10,  # Generate more questions
            no_repeat_ngram_size=2,
            top_k=50,  # Experiment with different values
            top_p=0.95,  # Experiment with different values
            do_sample=True,  # Enable sampling
            num_beams=10,  # Specify the number of beams (sequences) to generate
            temperature=0.7  # You can adjust the temperature
        )

        generated_question = question_tokenizer.decode(generated_question[0], skip_special_tokens=True)
        self.questions.append(generated_question)

    def get_user_answer(self, user_answer):
        # Get the user's answer to the current question
        is_correct = self.validate_user_answer(user_answer)  # Validate user's answer
        validation_prompt = self.create_validation_prompt(user_answer, is_correct)

        self.answers.append(validation_prompt)
        self.question_index += 1

    def create_validation_prompt(self, user_answer, is_correct):
        matching_similarity = self.calculate_matching_similarity(user_answer, self.answers[-1])

        validation_result = "Yes" if is_correct else "No"
        validation_prompt = f"User Answer: {user_answer}\nIs the user's answer correct? (Matching Similarity: {matching_similarity:.2f}) - {validation_result}\n"

        return validation_prompt

    def calculate_matching_similarity(self, answer1, answer2):
        embeddings1 = sentence_transformer_model.encode(answer1, convert_to_tensor=True)
        embeddings2 = sentence_transformer_model.encode(answer2, convert_to_tensor=True)
        cosine_similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
        return cosine_similarity.item()

    def validate_user_answer(self, user_answer):
        matching_similarity = self.calculate_matching_similarity(user_answer, self.answers[-1])
        return matching_similarity >= min_matching_similarity

# Create an instance of the chatbot
chatbot = QAChatbot()

app = FastAPI()

# Define the root endpoint
@app.get("/")
def root():
    return {"message": "Fast API in Python"}

# Upload a PDF and process it
@app.post("/LLM_marketplace/upload/")
async def upload_pdfs(pdf_files: List[UploadFile]):
    results = []

    for pdf_file in pdf_files:
        try:
            # Create a temporary file to store the uploaded PDF
            with NamedTemporaryFile(delete=False) as temp_pdf:
                temp_pdf.write(pdf_file.file.read())
                temp_pdf_path = temp_pdf.name

            # Save the temporary PDF file to S3
            s3.upload_file(temp_pdf_path, s3_bucket_name, pdf_file.filename)

            # Extract text from the PDF document
            loader = PyPDFLoader(temp_pdf_path)
            pages = loader.load_and_split()
            pdf_text = " ".join([page.page_content for page in pages])
            print("pdf_text is:")
            print(pdf_text)

            # Create document chunks
            chatbot.chunks = text_splitter.create_documents([pdf_text])

            results.append({"filename": pdf_file.filename, "message": "PDF uploaded to S3 and processed successfully"})
        except botocore.exceptions.NoCredentialsError:
            results.append({"filename": pdf_file.filename, "error": "AWS credentials not found. Make sure you've configured your AWS credentials."})
        finally:
            # Clean up the temporary PDF file
            os.remove(temp_pdf_path)

    return results


# Generate a question based on the current document chunk
@app.get("/LLM_marketplace/generate_question/")
async def generate_question():
    if chatbot.question_index < len(chatbot.chunks):
        chunk_text = chatbot.chunks[chatbot.question_index].page_content
        print("chunk_text is:")
        print(chunk_text)

        chatbot.generate_question(chunk_text)

        # Store the generated question in the Pinecone index
        pinecone.create_index(index_name, dimension=1536)

        pinecone.upsert_items(index_name, [{'ids': [chatbot.questions[-1]], 'vectors': [chatbot.questions[-1]]}])


        chatbot.question_index += 1  # Increment the question index

        return {"question": chatbot.questions[-1]}  # Return the generated question
    else:
        return {"message": "No more chunks to process"}  # Indicate that all chunks have been processed



# User answers
@app.post("/LLM_marketplace/user_answer/")
async def user_answer(answer: str):
    chatbot.get_user_answer(answer)

    if not chatbot.answers:
        return {"message": "No questions have been generated yet."}

    if chatbot.question_index >= len(chatbot.chunks):
        return {"message": "All questions have been answered."}

    return {
        "message": "Answer received successfully",
        "validation_prompt": chatbot.answers[-1],  # Return the validation prompt
    }

# Retrieve recorded answers
@app.get("/LLM_marketplace/get_answers/")
async def get_answers():
    if chatbot.answers:
        return {"answers": chatbot.answers}  # Return recorded answers
    else:
        return {"message": "No answers have been recorded yet"}  # Indicate that no answers have been recorded

# Run the FastAPI app using uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.1.1", port=8000)