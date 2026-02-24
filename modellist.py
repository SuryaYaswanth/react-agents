from groq import Groq
from dotenv import load_dotenv
import os
load_dotenv()

# Replace with your API key
client = Groq(api_key=os.environ["GROQ_API_KEY"])

models = client.models.list()

for model in models.data:
    print(model.id)