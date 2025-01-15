import tkinter as tk
from tkinter import scrolledtext
from chatbot_logic import get_bot_response

def send_message():
    user_input = user_entry.get()
    if user_input.strip() == "":
        return
    chat_display.insert(tk.END, f"User: {user_input}\n")
    user_entry.delete(0, tk.END)
    
    if user_input.lower() in ["exit", "quit", "bye"]:
        chat_display.insert(tk.END, "Chatbot: Goodbye! Keep practicing algebra!\n")
        root.quit()
        return

    bot_response = get_bot_response(user_input)
    chat_display.insert(tk.END, f"Chatbot: {bot_response}\n")

# GUI Code
root = tk.Tk()
root.title("Algebra Chatbot")
chat_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=20, bg="#f4f4f4", font=("Arial", 12))
chat_display.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
user_entry = tk.Entry(root, width=50, font=("Arial", 12))
user_entry.grid(row=1, column=0, padx=10, pady=10)
send_button = tk.Button(root, text="Send", command=send_message, bg="blue", fg="white", font=("Arial", 12))
send_button.grid(row=1, column=1, padx=10, pady=10)
root.mainloop()
