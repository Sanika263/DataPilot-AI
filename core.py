
import os
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

PROMPT = """
You are a data analyst. You receive a pandas DataFrame called df.
Write ONE line of pandas code that answers the user's question.
Save the answer in a variable called result.
Return code only. No explanations or markdown.

Example:
result = df.groupby('region')['sales'].sum()

Use only existing columns and pandas operations.
Do not use imports, file access, network access, or operating-system commands.
"""

def ask_question(df, question):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Check your .env file."
        )

    client = Groq(api_key=api_key)

    message = f"""
Columns: {list(df.columns)}

Data types:
{df.dtypes.to_string()}

Sample rows:
{df.head(3).to_string(index=False)}

Question: {question}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": message}
        ],
        temperature=0,
        max_completion_tokens=300
    )

    code = response.choices[0].message.content.strip()

    # Remove markdown code fences, if present
    if code.startswith("```"):
        lines = code.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines).strip()

    return code


def run_code(df, code):
    # Basic guardrails, not a complete security sandbox.
    forbidden = [
        "import ", "__", "open(", "exec(", "eval(",
        "compile(", "os.", "sys.", "subprocess",
        "socket", "requests.", "shutil."
    ]

    if any(term in code.lower() for term in forbidden):
        return "Error: Generated code contains a disallowed operation."

    data = {"df": df.copy()}

    try:
        exec(code, {"__builtins__": {}}, data)

        if "result" not in data:
            return "Error: AI did not create a result variable."

        return data["result"]

    except Exception as error:
        return f"Error: {error}"
