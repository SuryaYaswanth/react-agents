from dotenv import load_dotenv
import os
load_dotenv()


if __name__ == "__main__":
    print("Hello ReAct langgraph with Function calling")
    print(os.getenv("GEMINI_API_KEY"))
