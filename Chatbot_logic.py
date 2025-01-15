import os
import json
from sympy import symbols, Eq, solve
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application, parse_expr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re

def normalize_text(text):
    """Normalize text for consistent matching."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Predefined Responses
def get_predefined_response(user_query, file_path="algebra_data.json"):
    with open(file_path, "r") as f:
        predefined_responses = json.load(f)

    user_query_normalized = normalize_text(user_query)

    for entry in predefined_responses:
        if normalize_text(entry["prompt"]) == user_query_normalized:
            return entry["response"]

    return None

# SymPy Solver
def solve_algebra(problem):
    try:
        # Define the variable(s) used in the equation
        x = symbols('x')

        # Replace '^' with '**' for SymPy compatibility
        problem = problem.replace("^", "**").replace("Solve", "").strip()

        # Enable implicit multiplication and other transformations
        transformations = (standard_transformations + (implicit_multiplication_application,))

        # Parse the equation
        if "=" in problem:
            left, right = problem.split("=")
            equation = Eq(parse_expr(left.strip(), transformations=transformations), 
                          parse_expr(right.strip(), transformations=transformations))
        else:
            equation = parse_expr(problem.strip(), transformations=transformations)

        # Debugging: Log the parsed equation
        print(f"Debug: Parsed equation: {equation}")

        # Solve the equation
        solutions = solve(equation, x)

        # Debugging: Log the solutions
        print(f"Debug: solutions type={type(solutions)}, value={solutions}")

        # Handle different types of solutions
        if isinstance(solutions, list):
            if len(solutions) == 0:
                return "No solutions found."
            else:
                return f"The solutions are: {', '.join(map(str, solutions))}"
        elif isinstance(solutions, (int, float, symbols)):  # Single solution
            return f"The solution is: {solutions}"
        else:
            # Handle unexpected types
            return f"Unexpected solution type: {type(solutions)} - {solutions}"

    except Exception as e:
        # Debugging: Print the error
        print(f"Debug: Error in solve_algebra: {e}")
        return f"Sorry, I couldn't solve that problem. Error: {e}"


# Load Pretrained Model and Tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Add a padding token if not present
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))  # Resize the embeddings

chatbot = pipeline("text-generation", model="./algebra_chatbot", tokenizer="./algebra_chatbot")

# Unified Function to Get Bot Response
def get_bot_response(user_input):
    print(f"Debug: User input: {user_input}")

    # Check for predefined responses
    predefined_response = get_predefined_response(user_input)
    if predefined_response:
        print(f"Debug: Predefined response: {predefined_response}")
        return predefined_response

    # Check for algebraic problems
    if "solve" in user_input.lower() or "=" in user_input:
        result = solve_algebra(user_input)
        print(f"Debug: solve_algebra result: {result}")
        return result

    # Use the fine-tuned model for general questions
    else:
        response = chatbot(
            f"User: {user_input}\nBot:",
            max_length=100,
            num_return_sequences=1,
            truncation=True
        )
        bot_response = response[0]["generated_text"].split("Bot:")[-1].strip()
        print(f"Debug: chatbot response: {bot_response}")
        return bot_response

   
