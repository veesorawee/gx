# Automated Data Validation Pipeline

This Streamlit application provides a user-friendly interface for running data validation tests using Great Expectations against a Trino database. It features two modes: a Manual mode for running custom queries and expectations, and an Automate mode that leverages Google's Gemini API to generate expectations from natural language specifications.

## About This Project

This application was developed through an iterative and collaborative process between a human developer and several large language models, primarily **Google's Gemini**, with contributions from **Anthropic's Claude** and **OpenAI's ChatGPT**.

The AI models served as powerful coding assistants, providing:
* Code generation for functions and UI components.
* Complex logic and algorithm design, especially for the "Automate Mode".
* Advanced "Prompt Engineering" to generate precise Great Expectations rules.
* Iterative debugging and error resolution.

This project stands as a testament to the power of human-AI collaboration in modern software development.

## Features

- **Dual Modes:** Choose between a fully manual workflow or an AI-assisted automate mode.
- **Dynamic SQL Generation:** Automate mode can create and test SQL queries based on simple target inputs.
- **AI-Powered Expectation Generation:** Use natural language or a spec table to generate complex Great Expectations rules with Gemini.
- **File Management:** Load, preview, and save SQL queries and Expectation Suites directly from the UI.
- **Interactive Results:** View validation results, error summaries with percentages, and data samples within the app.
- **Data Docs Integration:** Automatically build and open Great Expectations Data Docs for in-depth analysis.

## Setup Instructions

### Prerequisites

* **Python 3.9+**
* **Git**

### Installation for macOS (Apple Silicon: M1, M2, M3)

These steps ensure that all dependencies, especially those requiring compilation, are installed correctly on Apple's architecture.

1.  **Install Homebrew:** If you don't have it, open your Terminal and install it. It's the standard package manager for macOS.
    ```bash
    /bin/bash -c "$(curl -fsSL [https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh](https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh))"
    ```

2.  **Install Xcode Command Line Tools:** Many Python packages need this to compile correctly.
    ```bash
    xcode-select --install
    ```

3.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-project-folder>
    ```

4.  **Create and Activate a Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

5.  **Install Dependencies:** Use the provided `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

### General Configuration (All Operating Systems)

1.  **Set Up Environment Variables:**
    * Copy the example file: `cp .env.example .env`
    * Edit the `.env` file and fill in your actual database credentials and your Google Gemini API Key.
    
2.  **Prepare Template Files (Optional):**
    * The `prompt_template.txt` and `sql_template.txt` files are required for Automate Mode. Ensure they are present in your project root.

3.  **Prepare Directories:**
    * The application will automatically create the `sql/` and `expectations/` directories if they don't exist. You can add your own `.sql` and `.json` files to these folders to have them appear in the UI.

## How to Run

Once the setup is complete, run the following command in your terminal (make sure your virtual environment is activated):

```bash
streamlit run app.py
```

The application will open in your default web browser.

## Project Structure

```
.
├── .env                # Your local environment variables (created from .env.example)
├── .env.example        # Example environment file
├── .gitignore          # Tells Git which files to ignore
├── app.py              # The main Streamlit application code
├── expectations/       # Directory to store expectation .json files
├── prompt_template.txt # The prompt template for Gemini
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── sql/                # Directory to store .sql files
└── sql_template.txt    # The base template for auto-generated SQL
```