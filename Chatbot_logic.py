import sympy as sp
import re
import json

def preprocess_equation(equation_str):
    """Preprocess the equation to handle implicit multiplication."""
    # Add a '*' between a number and a variable (e.g., '2x' -> '2*x')
    equation_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', equation_str)
    return equation_str

def solve_equation(equation_str, variable):
    """Solve the given algebraic equation for the specified variable."""
    try:
        # Preprocess the equation to handle implicit multiplication
        equation_str = preprocess_equation(equation_str)

        # Normalize the equation string by ensuring proper spacing around '='
        equation_str = equation_str.replace("=", " = ")
        parts = equation_str.split('=')
        if len(parts) != 2:
            return "Error: The equation must have exactly one '=' symbol."
        lhs, rhs = parts
        eq = sp.Eq(sp.sympify(lhs.strip()), sp.sympify(rhs.strip()))

        # Solve the equation
        solution = sp.solve(eq, sp.Symbol(variable))
        if not solution:
            return "No solutions"
        return solution
    except Exception as e:
        return f"Error: {e}"

def simplify_expression(expression_str):
    """Simplify the given algebraic expression."""
    try:
        # Preprocess the expression to handle implicit multiplication
        expression_str = preprocess_equation(expression_str)

        # Simplify the expression
        simplified = sp.simplify(sp.sympify(expression_str))
        return simplified
    except Exception as e:
        return f"Error: {e}"

def load_prompts(file_path):
    """Load predefined prompts and responses from a JSON file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def respond_to_prompt(user_input, prompts):
    """Check if user input matches any predefined prompt and return a response."""
    for item in prompts:
        if item['prompt'].lower() in user_input.lower():
            return item['response']
    return None

def chatbot():
    """Run the algebra-solving chatbot."""
    prompts = load_prompts('algebra_data.json')
    print("Welcome to the Algebra Solving Chatbot!")
    print("You can ask me to solve equations, simplify expressions, or general questions about algebra.")
    print("Type 'exit' to leave the chatbot.")

    while True:
        user_input = input("\nEnter your question: ").strip()

        if user_input.lower() == 'exit':
            print("Goodbye! Have a great day.")
            break

        # Check for predefined prompts
        prompt_response = respond_to_prompt(user_input, prompts)
        if prompt_response:
            print(prompt_response)
            continue

        if 'solve' in user_input.lower():
            try:
                # Extract the equation and variable from the input
                print("For example, to solve '2*x + 3 = 7' for x, type: solve 2*x + 3 = 7 for x")
                equation_part = user_input.split('solve', 1)[1].strip()
                if 'for' not in equation_part:
                    print("Error: Please specify the variable using 'for'.")
                    continue
                equation, variable_part = equation_part.rsplit('for', 1)
                variable = variable_part.strip()

                # Solve the equation
                solution = solve_equation(equation.strip(), variable)
                print(f"Solution: {solution}")
            except Exception as e:
                print(f"Error: {e}")
        elif 'simplify' in user_input.lower():
            try:
                # Extract the expression from the input
                print("For example, to simplify '2*x + 3*x', type: simplify 2*x + 3*x")
                expression = user_input.split('simplify', 1)[1].strip()

                # Simplify the expression
                simplified = simplify_expression(expression)
                print(f"Simplified Expression: {simplified}")
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("I didn't understand that. Please ask me to solve or simplify an expression.")

if __name__ == "__main__":
    chatbot()
