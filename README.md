# Automated Data Validation Pipeline

This Streamlit application provides a user-friendly interface for running data validation tests using Great Expectations against a Trino database. It features two modes: a Manual mode for running custom queries and expectations, and an Automate mode that leverages Google's Gemini API to generate expectations from natural language specifications.

## Features

- **Dual Modes:** Choose between Manual and Automate workflows.
- **Dynamic SQL Generation:** Automate mode can create SQL queries based on simple target inputs.
- **AI-Powered Expectation Generation:** Use natural language or a spec table to generate complex Great Expectations rules with Gemini.
- **File Management:** Load, preview, and save SQL queries and Expectation Suites directly from the UI.
- **Interactive Results:** View validation results, error summaries, and data samples within the app.
- **Data Docs Integration:** Automatically build and open Great Expectations Data Docs for in-depth analysis.

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd your-validation-project
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    -   Copy the example `.env.example` file to a new file named `.env`.
    -   `cp .env.example .env`
    -   Edit the `.env` file and fill in your actual database credentials and your Google Gemini API Key.

5.  **Create necessary directories:**
    - The application will automatically create `sql/` and `expectations/` directories on first run, but you can create them manually if you wish.

## How to Run

Execute the following command in your terminal:

```bash
streamlit run app.py