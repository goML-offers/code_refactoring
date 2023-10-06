import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os


# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key here
OPEN_API_KEY = os.getenv('OPEN_API_KEY')

# Initialize your OpenAI API key
openai.api_key = OPEN_API_KEY

# Initialize FastAPI app
app = FastAPI()

# Functions

def get_response(prompt):
    '''
    Get the ChatGPT response for a given prompt using the provided API key.
    '''
    api_key = OPEN_API_KEY

    completion = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.2,
        api_key=api_key
    )

    return completion['choices'][0]['message']['content']

# FastAPI Application

app = FastAPI()

class InputData(BaseModel):
    code: str

@app.get("/")
def root():
    return {"message": "Fast API in Python"}

# Define an endpoint to process code
@app.post("/LLM marketplace/process_code/")
async def process_code(code: str):
    '''
    Process the provided code using ChatGPT.
    '''
    if not code:
        raise HTTPException(status_code=400, detail="Code not provided in the request")

    prompt = text + code
    response = get_response(prompt)
    return response


# Constants

text = '''
Please refactor the provided Python code snippet to incorporate the following changes:

1. Properly add comments to explain the code when needed.
2. Follow naming conventions for variables and functions and function parameters.
3. Add appropriate type annotations to function parameters and return types.
4. Include docstrings for functions.
5. Organize imports in a clean and orderly manner and remove unnecessary modules.
6. Remove any unnecessary extra lines and spaces.
7. Use f-strings wherever required

Below is the code snippet:

'''

# Run the FastAPI app using uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)