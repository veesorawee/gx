# Data Validation Pipeline with Great Expectations

A friendly Streamlit app that makes data validation super easy! 🚀

## What does this do?

This app helps you validate your data using Great Expectations and connects to your Trino database. You can either:
- **Manual Mode**: Write your own SQL queries and validation rules
- **Automate Mode**: Just describe what you want to check in plain English, and let AI (Google Gemini) generate the validation rules for you!

The app can automatically segment your data and run validations across different groups, making it perfect for checking data quality across multiple dimensions.

## Cool AI Story 🤖

This project was built with the help of AI assistants - mainly **Google's Gemini**, plus some help from **Claude** and **ChatGPT**. They helped with coding, debugging, and even writing the smart prompts that make the automation work. 

## What can it do?

- 🔄 **Two modes**: Pick manual control or let AI help you out
- 🗄️ **Database ready**: Connects to Trino databases with secure authentication
- 🔍 **Smart SQL generation**: AI can write SQL queries based on your app events
- 💬 **Plain rules**: Describe validation rules in normal language
- 📊 **Auto-segmentation**: Automatically discover and validate data segments
- 📁 **File management**: Load, save, and preview your SQL and validation files
- 📈 **Rich results**: See validation results with error percentages and data samples
- 📚 **Data Docs**: Automatically creates detailed HTML reports you can browse
- � **PDF Reports**: Auto-generates professional "Tracking Validation" PDF summaries.
- 🛑 **Waive Rules**: Exclude false positives, regenerate metrics, and update reports on the fly.
- �🔄 **Retry logic**: AI generation with smart fallbacks if something goes wrong

## Getting Started

### What you'll need first
- Python 3.9 or newer
- Git
- A Trino database to connect to
- Google Gemini API key (for Automate Mode)

### Easy Setup (Mac with Apple Silicon - M1/M2/M3)

Don't worry, we'll walk through this step by step!

**1. Get Homebrew (if you don't have it)**
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**2. Download the code**
```bash
git clone https://github.com/veesorawee/gx.git
cd gx
```

**3. Install OS Dependencies (for PDF Generation)**
```bash
brew install pango libffi glib
```

**4. Set up your Python environment**
```bash
# Create a clean space for this project
python3 -m venv venv

# Activate it (you'll need to do this each time)
source venv/bin/activate
```

**5. Install everything**
```bash
pip install -r requirements.txt
```

### Setup for Everyone

**1. Configure your settings**
```bash
# Copy the example file
cp .env.example .env

# Edit it with your details
nano .env  # or use any text editor you like
```

Fill in your details:
- **Database connection**: Trino host, port, username, and password
- **Google Gemini API key**: Get one from [Google AI Studio](https://aistudio.google.com/) (needed for Automate Mode)

**2. Check your files**
The app needs these template files (they should already be there):
- `prompt_template.txt` - Tells the AI how to generate validation rules
- `sql_template.txt` - Template for auto-generated SQL queries

**3. You're ready!**
The app will create these folders automatically when you run it:
- `sql/` - For your SQL query files
- `expectations/` - For your validation rule files
- `logs/` - For debugging AI generation

## Running the App

Make sure your virtual environment is active:
```bash
source venv/bin/activate  # if you're not already in it
```

Then start the app:
```bash
streamlit run app.py
```

Your browser should open automatically to the app!

## How to Use

### Manual Mode
1. Write or select a SQL query
2. Create JSON validation rules using Great Expectations syntax
3. Hit "Start Validation" and see the results!

### Automate Mode
1. **Choose your SQL source:**
   - Generate from target inputs (app events like clicks, impressions)
   - Select an existing SQL file
   - Write custom SQL manually

2. **Describe your validation needs** in plain English, like:
   - "Check that all user_ids are not null"
   - "Validate that event_timestamp is within the last 24 hours"
   - "Ensure click_count is always a positive number"

3. **Let AI do the work** - it will generate proper validation rules and run them!

### Pro Tips
- Use segmentation to validate data across different groups (like by country, device type, etc.)
- The app saves your SQL and validation files so you can reuse them
- Data Docs gives you beautiful HTML reports with all the details
- If AI generation fails, it automatically tries fallbacks

## What's in the folder?

```text
gx/
├── .env                # Your secret settings (don't share this!)
├── .env.example        # Example of what .env should look like
├── app_v6.py           # The main app code
├── expectations/       # Where validation rules are saved
├── logs/               # Debugging logs (auto-created)
├── reports/            # Output folder for generated PDF reports
├── prompt_template.txt # How the AI knows what to do
├── requirements.txt    # List of Python packages needed
├── sql/                # Where your SQL files live
└── sql_template.txt    # Template for auto-generated SQL
```

## Need Help?

If you run into issues:
1. Make sure your virtual environment is activated
2. Double-check your `.env` file has the right database credentials
3. For Automate Mode, ensure you have a valid Google Gemini API key
4. Check the `logs/` folder for detailed error messages from AI generation

Happy validating! 🎉