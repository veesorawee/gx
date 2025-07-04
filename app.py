import os
import urllib
import json
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy.engine import create_engine
from sqlalchemy import text
import pandas as pd
import great_expectations as gx
import datetime
import glob
import re
import traceback
import google.generativeai as genai
import webbrowser

# --- 0. App Configuration & Setup ---
SQL_DIR = "sql"
EXPECTATIONS_DIR = "expectations"
os.makedirs(SQL_DIR, exist_ok=True)
os.makedirs(EXPECTATIONS_DIR, exist_ok=True)

load_dotenv()

GEMINI_AVAILABLE = False
api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
if api_key:
    try:
        genai.configure(api_key=api_key)
        GEMINI_AVAILABLE = True
    except Exception:
        GEMINI_AVAILABLE = False

# --- 1. Helper Functions ---
def list_files(directory, extension):
    return [os.path.basename(f) for f in glob.glob(os.path.join(directory, f"*.{extension}"))]

def flatten_json_columns(df: pd.DataFrame, flatten_map: dict) -> pd.DataFrame:
    for parent_col, keys_to_flatten in flatten_map.items():
        if parent_col not in df.columns or not keys_to_flatten: continue
        def safe_get_from_json_string(json_string, key):
            if isinstance(json_string, str):
                try: return json.loads(json_string).get(key)
                except json.JSONDecodeError: return None
            return None
        for key in keys_to_flatten:
            df[f'{parent_col}_{key}'] = df[parent_col].apply(safe_get_from_json_string, key=key)
    return df

def generate_automated_sql(screen_name: str, app_type: str, event_type: str, event_category: str, target: str) -> str:
    """Generates SQL with a dynamic WHERE clause based on event_type."""
    table_name_middle = f"_{app_type}" if app_type == "web" else ""
    table_name = f"jitsu_lm_user{table_name_middle}_{event_type}"
    
    try:
        with open("sql_template.txt", "r", encoding="utf-8") as f:
            base_query_template = f.read()
    except FileNotFoundError:
        st.error("File not found: 'sql_template.txt'. Please create it first.")
        st.stop()

    # Step 1: Fill in the table name from the template
    base_query = base_query_template.format(table_name=table_name)
    
    # Step 2: Build the WHERE clause dynamically
    where_clauses = [f"dt = '{{dt}}'"] # Keep dt as a placeholder for the main process
    if screen_name:
        where_clauses.append(f"screen_name = '{screen_name}'")

    if event_type in ["click", "impression"]:
        if event_category: where_clauses.append(f"event_category = '{event_category}'")
        if target: where_clauses.append(f"target = '{target}'")
    elif event_type == "custom":
        if event_category: where_clauses.append(f"event_category = '{event_category}'")
    
    # Step 3: Combine base query with the dynamic WHERE clause
    final_template = f"{base_query} WHERE {' AND '.join(where_clauses)} {{extra_where_conditions}}"
    return final_template

def display_validation_errors(checkpoint_result):
    failed_results_list = []
    if not checkpoint_result.success:
        for result_identifier, validation_result_dict in checkpoint_result.run_results.items():
            batch_id = result_identifier.batch_identifier
            match = re.search(r'pandas_datasource_(.*?)-asset_', batch_id)
            segment = match.group(1) if match else batch_id
            validation_result = validation_result_dict['validation_result']
            for result in validation_result.results:
                if not result.success:
                    kwargs = result.expectation_config.kwargs
                    expected = kwargs.get("value_set") or kwargs.get("regex") or kwargs.get("sum_total") or kwargs.get("type_")
                    unexpected_count = result.result.get("unexpected_count", 0)
                    element_count = result.result.get("element_count", 0)
                    error_percent = (unexpected_count / element_count) * 100 if element_count > 0 else 0
                    failed_results_list.append({
                        "Segment": segment, "Expectation": result.expectation_config.expectation_type,
                        "Column": kwargs.get("column", "N/A"), "Expected": str(expected),
                        "Error Count": unexpected_count, "Error %": error_percent,
                        "Failing Examples": result.result.get("partial_unexpected_list", [])
                    })
    if failed_results_list:
        st.subheader("Failure Summary")
        df = pd.DataFrame(failed_results_list)
        st.data_editor(df, column_config={
                "Error %": st.column_config.ProgressColumn("Error Rate", format="%.2f%%", min_value=0, max_value=100),
                "Failing Examples": st.column_config.ListColumn("Examples", width="large"),
                "Expected": st.column_config.TextColumn("Expected Value", width="medium"),
            }, use_container_width=True, hide_index=True)


def parse_select_columns(sql_string: str) -> list[str]:
    """Parses a SQL string to extract column names from the main SELECT clause."""
    try:
        # This regex finds the content between SELECT and FROM, ignoring case and newlines
        match = re.search(r'SELECT(.*?)FROM', sql_string, re.IGNORECASE | re.DOTALL)
        if not match:
            return []
        
        # Clean up, split by comma, and strip whitespace from each column
        columns_str = match.group(1).replace('\n', ' ').replace('\r', '')
        columns = [col.strip().split('.')[-1].strip() for col in columns_str.split(',')]
        return [col for col in columns if col]
    except Exception:
        return []


def run_validation_process(sql_input, expectations_input, suite_name, db_config, val_params):
    trino_host, trino_port, trino_username, trino_password = db_config
    db_query_date, SEGMENT_BY_COLUMNS = val_params

    with st.status("🚀 Starting Validation Process...", expanded=True) as status:
        run_name = f"run_{suite_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        status.write(f"**Run Name:** `{run_name}` | **Suite:** `{suite_name}`")

        try:
            expectations_list = json.loads(expectations_input)
        except json.JSONDecodeError:
            status.update(label="Input Error", state="error"); st.error("Invalid Expectations JSON format."); return None, None
        
        status.write("🔍 Discovering object columns and fields to flatten from expectations...")
        
        # 1. Get all top-level columns from the SQL to avoid flattening existing columns
        top_level_columns = parse_select_columns(sql_input)
        
        all_columns_in_rules = {exp['kwargs']['column'] for exp in expectations_list if 'column' in exp.get('kwargs', {})}
        
        # 2. Infer parent objects by looking for the 'parent_child' naming convention
        potential_parents = {col.split('_')[0] for col in all_columns_in_rules if '_' in col}
        object_columns = sorted(list(potential_parents))

        keys_to_flatten_map = {obj_col: [] for obj_col in object_columns}
        
        if object_columns:
            status.write(f"**Identified potential Object columns:** `{object_columns}`")
            for col_name in all_columns_in_rules:
                # 3. Check if a rule's column is a child of a potential parent
                if "_" in col_name and col_name.split('_')[0] in object_columns:
                    # 4. CRITICAL: Only flatten if the column does NOT already exist at the top level
                    if col_name not in top_level_columns:
                        parent = col_name.split('_')[0]
                        key = col_name[len(parent) + 1:]
                        if key not in keys_to_flatten_map[parent]:
                            keys_to_flatten_map[parent].append(key)

            status.write("🗺️ **Final fields to flatten:**")
            for obj_col, keys in keys_to_flatten_map.items():
                if keys: status.write(f"&nbsp;&nbsp;&nbsp;↳ For `{obj_col}`: `{sorted(keys)}`", unsafe_allow_html=True)
        
        status.write(f"**1. 🔗 Connecting to DB...**")
        try:
            engine = create_engine(f'trino://{trino_username}:{urllib.parse.quote(trino_password)}@{trino_host}:{trino_port}/', connect_args={'http_scheme': 'https', 'source': 'gx-streamlit-app'})
            with engine.connect() as conn: conn.execute(text("SELECT 1"))
            status.write("... ✅ Connection successful.")
        except Exception as e:
            status.update(label="Connection Failed", state="error"); st.error(f"Could not connect to Database: {e}"); return None, None
        
        segments_to_run = []
        if SEGMENT_BY_COLUMNS:
            status.write(f"**2. 🔭 Discovering segments by:** `{', '.join(SEGMENT_BY_COLUMNS)}`")
            try:
                base_query_for_discovery = sql_input.format(dt=db_query_date, extra_where_conditions="")
                discovery_sql = f"SELECT {', '.join(SEGMENT_BY_COLUMNS)} FROM ({base_query_for_discovery}) AS discovery_subquery GROUP BY {', '.join(SEGMENT_BY_COLUMNS)}"
                segments_df = pd.read_sql(text(discovery_sql), engine)
                segments_to_run = segments_df.to_dict('records')
            except Exception as e:
                status.update(label="Error", state="error"); st.error(f"Error discovering segments: {e}"); return None, None
            if not segments_to_run:
                status.update(label="Warning", state="error"); st.warning(f"⚠️ No segments found for date `{db_query_date}`."); return None, None
            status.write(f"... ✅ Found **{len(segments_to_run)}** segments.")
        else:
            status.write("**2. 🗂️ No segmentation specified.**")
            segments_to_run = [{}] 
        
        with st.expander("View Discovered Segments"):
            st.json(segments_to_run)

        context = gx.get_context(); context.add_or_update_expectation_suite(suite_name); validations_to_run = []
        status.write("**3. ⚙️ Processing each segment...**")
        progress_bar = st.progress(0, text="Processing segments...")
        
        data_samples = {}
        for i, segment_filters in enumerate(segments_to_run):
            segment_name = "-".join(str(v) for v in segment_filters.values()).replace(".", "_") if segment_filters else "full_dataset"
            status.write(f"&nbsp;&nbsp;&nbsp;↳ **Segment {i+1}/{len(segments_to_run)}:** `{segment_name}`", unsafe_allow_html=True)
            extra_where_conditions = " ".join([f"AND {col} = '{val}'" for col, val in segment_filters.items()])
            final_query = sql_input.format(dt=db_query_date, extra_where_conditions=extra_where_conditions) + " LIMIT 500"
            try:
                df = pd.read_sql(text(final_query), engine)
                if df.empty: status.write(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- No data. Skipping.", unsafe_allow_html=True); continue
                
                # 1. Standardize object columns to string first.
                for col_name in object_columns:
                    if col_name in df.columns: df[col_name] = df[col_name].apply(lambda x: json.dumps(x) if x is not None else None)
                
                # 2. Flatten the dataframe. It now contains all columns.
                df = flatten_json_columns(df, keys_to_flatten_map)
                
                # 3. Store a sample of THIS final, flattened dataframe
                data_samples[segment_name] = df.head(5)
                with st.expander(f"🔬 Data Sample (First 5 Rows) for Segment: {segment_name}"):
                    st.dataframe(data_samples[segment_name], use_container_width=True)

                # 4. Build the batch request from the SAME flattened dataframe
                datasource = context.sources.add_or_update_pandas(f"pandas_datasource_{segment_name}")
                asset_name = f"asset_{segment_name}"
                data_asset = datasource.add_dataframe_asset(name=asset_name)
                batch_request = data_asset.build_batch_request(dataframe=df) # Use the final df
                
                # 5. The validator now receives the correct data
                validator = context.get_validator(batch_request=batch_request, expectation_suite_name=suite_name)
                for exp_config in expectations_list:
                    getattr(validator, exp_config['expectation_type'])(**exp_config["kwargs"])
                
                validator.save_expectation_suite(discard_failed_expectations=False)
                validations_to_run.append({"batch_request": batch_request, "expectation_suite_name": suite_name})
            except Exception as e:
                tb_str = traceback.format_exc()
                status.write(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- 🚨 Failed: {e}", unsafe_allow_html=True)
                with st.expander(f"Full Error Traceback for Segment: {segment_name}"):
                    st.code(tb_str, language='text')
                continue
            progress_bar.progress((i + 1) / len(segments_to_run), text=f"Processed segment: {segment_name}")
            
        if not validations_to_run: 
            status.update(label="Error", state="error"); st.error("No valid batches were created."); return None, None
        
        status.write("**4. 🏁 Running final Great Expectations checkpoint...**")
        checkpoint = context.add_or_update_checkpoint(name=f"checkpoint_{run_name}", run_name_template=run_name)
        checkpoint_result = checkpoint.run(validations=validations_to_run)
        
        status.update(label="✅ **Validation Process Complete!**", state="complete", expanded=False)
    
    return checkpoint_result, data_samples

# --- 2. Main App UI ---
st.set_page_config(layout="wide", page_title="Automated Data Validation")
st.title("🚀 Automated Data Validation Pipeline")

with st.sidebar:
    st.header("⚙️ Database Configuration")
    trino_host_env = os.getenv('TRINO_HOST')
    trino_port_env = os.getenv('TRINO_PORT')
    trino_user_env = os.getenv('TRINO_USERNAME')
    trino_pass_env = os.getenv('TRINO_PASSWORD')

    if all([trino_host_env, trino_port_env, trino_user_env, trino_pass_env]):
        st.success("✅ Loaded DB credentials from .env")
        trino_host = trino_host_env
        trino_port = trino_port_env
        trino_username = trino_user_env
        trino_password = trino_pass_env
    else:
        st.info("Some DB credentials not found in .env. Please provide them below.")
        trino_host = st.text_input("Trino Host", value=trino_host_env or "iu-trino-adhoc.linecorp.com")
        trino_port = st.text_input("Trino Port", value=trino_port_env or "8443")
        trino_username = st.text_input("Trino Username", value=trino_user_env or "")
        trino_password = st.text_input("Trino Password", type="password", value=trino_pass_env or "")

    st.header("📅 Validation Parameters")
    yesterday = datetime.date.today() - datetime.timedelta(days=1)
    db_query_date = st.text_input("Query Date (YYYY-MM-DD)", value=yesterday.strftime('%Y%m%d'))
    segment_by_columns_str = st.text_input("Columns to Segment By", help="Comma-separated. Leave empty to validate all data as one batch.")
    SEGMENT_BY_COLUMNS = [col.strip() for col in segment_by_columns_str.split(',') if col.strip()]

manual_tab, automate_tab = st.tabs(["Manual Mode", "Automate Mode"])

with manual_tab:
    st.header("Manual Mode Configuration")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📄 SQL Query")
        sql_files = list_files(SQL_DIR, "sql")
        sql_file_options = ["--- Create New ---"] + sql_files
        selected_sql_file = st.selectbox("Select existing SQL file or create new:", sql_file_options, key="manual_sql_selector")
        sql_content = ""
        if selected_sql_file != "--- Create New ---":
            with open(os.path.join(SQL_DIR, selected_sql_file), 'r', encoding='utf-8') as f: sql_content = f.read()
        manual_sql_input = st.text_area("SQL Query Template:", value=sql_content, height=350, key="manual_sql_text_area")
        if selected_sql_file == "--- Create New ---":
            new_sql_filename = st.text_input("New SQL Filename:", key="manual_new_sql_name", placeholder="my_query.sql")
            if st.button("Save New SQL File", key="manual_save_sql"):
                if new_sql_filename and new_sql_filename.endswith(".sql"):
                    with open(os.path.join(SQL_DIR, new_sql_filename), 'w', encoding='utf-8') as f: f.write(manual_sql_input)
                    st.success(f"Saved file '{new_sql_filename}'!"); st.rerun()
                else: st.error("Please provide a filename ending with .sql")
    with col2:
        st.subheader("Expectation Rules (JSON)")
        exp_files = list_files(EXPECTATIONS_DIR, "json")
        exp_file_options = ["--- Create New ---"] + exp_files
        selected_exp_file = st.selectbox("Select existing Expectation file or create new:", exp_file_options, key="manual_exp_selector")
        exp_content = ""
        if selected_exp_file != "--- Create New ---":
            with open(os.path.join(EXPECTATIONS_DIR, selected_exp_file), 'r', encoding='utf-8') as f: exp_content = f.read()
        manual_expectations_input = st.text_area("Expectations JSON:", value=exp_content, height=350, key="manual_exp_text_area")
        if selected_exp_file == "--- Create New ---":
            new_exp_filename = st.text_input("New Expectation Filename:", key="manual_new_exp_name", placeholder="my_rules.json")
            if st.button("Save New Expectation File", key="manual_save_exp"):
                if new_exp_filename and new_exp_filename.endswith(".json"):
                    try:
                        json.loads(manual_expectations_input)
                        with open(os.path.join(EXPECTATIONS_DIR, new_exp_filename), 'w', encoding='utf-8') as f: f.write(manual_expectations_input)
                        st.success(f"Saved file '{new_exp_filename}'!"); st.rerun()
                    except json.JSONDecodeError: st.error("The content is not valid JSON.")
                else: st.error("Please provide a filename ending with .json")
    
    st.markdown("---")
    if st.button("🚀 Start Validation (Manual)", type="primary"):
        if SEGMENT_BY_COLUMNS and "{extra_where_conditions}" not in manual_sql_input:
            st.error("Segmentation is enabled, but the required '{extra_where_conditions}' placeholder is missing in your SQL query.")
            st.stop()
        suite_name = os.path.splitext(selected_exp_file)[0] if selected_exp_file != "--- Create New ---" else "manual_suite"
        db_config = (trino_host, trino_port, trino_username, trino_password)
        val_params = (db_query_date, SEGMENT_BY_COLUMNS)
        checkpoint_result, data_samples = run_validation_process(manual_sql_input, manual_expectations_input, suite_name, db_config, val_params)
        
        if checkpoint_result:
            st.header("🔬 Data Samples (First 5 Rows)")
            if data_samples:
                for segment_name, sample_df in data_samples.items():
                    with st.expander(f"Data for Segment: {segment_name}"):
                        st.dataframe(sample_df, use_container_width=True)


            st.header("📊 Final Validation Results")
            if checkpoint_result.success: st.success("🎉 **Overall Status: SUCCESS**")
            else: st.error("🚨 **Overall Status: FAILURE**")
            display_validation_errors(checkpoint_result)

            with st.expander("View Full JSON Result"):
                st.json(checkpoint_result.to_json_dict())
            
            st.subheader("🔗 Data Docs")
            with st.spinner("Building Data Docs..."):
                context = gx.get_context(); context.build_data_docs()
                docs_sites = context.get_docs_sites_urls()
                if docs_sites:
                    st.info("Copy the path below into a new browser tab to open the full report:")
                    st.code(docs_sites[0]['site_url'], language=None)
                    webbrowser.open(docs_sites[0]['site_url'])
                else:
                    st.warning("Could not generate a path to Data Docs.")


with automate_tab:
    st.header("Automate Mode Configuration")
    st.markdown("Choose a source for your SQL query, then provide a data spec.")
    
    sql_source = st.radio("1. Choose SQL Source", ["Generate from Target Inputs", "Select Existing SQL File", "Write Custom SQL Manually"], key="auto_sql_source_selector", horizontal=True)

    if sql_source == "Generate from Target Inputs":
        with st.container(border=True):
            st.subheader("Define Target for SQL Generation")
            row1_cols = st.columns(2)
            with row1_cols[0]:
                auto_app_type = st.selectbox("App Type", ["native", "app", "web"], key="auto_app_type_gen")
            with row1_cols[1]:
                auto_event_type = st.selectbox("Event Type", ["click", "impression", "pageview", "custom"], key="auto_event_type_gen")
            
            row2_cols = st.columns(3)
            with row2_cols[0]:
                auto_screen_name = st.text_input("Screen Name", key="auto_screen_name_gen")
            if auto_event_type in ["click", "impression", "custom"]:
                with row2_cols[1]:
                    auto_event_category = st.text_input("Event Category", key="auto_event_category_gen")
            if auto_event_type in ["click", "impression"]:
                with row2_cols[2]:
                    auto_target = st.text_input("Target", key="auto_target_gen")
    
    elif sql_source == "Select Existing SQL File":
        with st.container(border=True):
            st.subheader("Select Target SQL File")
            sql_files = list_files(SQL_DIR, "sql")
            if not sql_files:
                st.warning(f"No SQL files found in the '{SQL_DIR}/' directory.")
            else:
                selected_manual_sql = st.selectbox("Select from existing SQL files:", [""] + sql_files, key="auto_sql_file_selector")
                if selected_manual_sql:
                    with open(os.path.join(SQL_DIR, selected_manual_sql), 'r', encoding='utf-8') as f:
                        st.code(f.read(), language="sql", line_numbers=True)
    
    else: # Write Custom SQL Manually
        with st.container(border=True):
            st.subheader("Enter Custom SQL Template")
            st.info("For segmentation, your query must include `{extra_where_conditions}`.")
            custom_sql_input = st.text_area("SQL Query Input:", key="auto_custom_sql_input", height=200)
            if custom_sql_input:
                st.write("SQL Preview:")
                st.code(custom_sql_input, language="sql", line_numbers=True)

    st.subheader("2. Provide Data Spec")
    if GEMINI_AVAILABLE:
        auto_spec_input = st.text_area("Description or Spec Table:", height=250, key="auto_spec_input", help="Enter complex rules and the data spec table here.")
    else:
        st.warning("Gemini API Key not found. Please set it up in .env or st.secrets to enable this feature.")
        auto_spec_input = ""

    st.markdown("---")
    if st.button("🚀 Generate & Start Validation (Automate)", type="primary"):
        final_sql_input = ""
        suite_name_prefix = "auto"
        
        # Step 1: Determine which SQL to use
        selected_sql_source = st.session_state.auto_sql_source_selector
        if selected_sql_source == "Generate from Target Inputs":
            event_type = st.session_state.auto_event_type_gen
            screen_name = st.session_state.auto_screen_name_gen
            event_category = st.session_state.get("auto_event_category_gen", "")
            target = st.session_state.get("auto_target_gen", "")
            if not screen_name or (event_type in ["click", "impression"] and not all([event_category, target])) or (event_type == "custom" and not event_category):
                st.error("Please fill in all required fields for the selected Event Type."); st.stop()
            final_sql_input = generate_automated_sql(screen_name, st.session_state.auto_app_type_gen, event_type, event_category, target)
            suite_name_prefix = f"{screen_name}_{event_category}_{target}"
        elif selected_sql_source == "Select Existing SQL File":
            selected_file = st.session_state.auto_sql_file_selector
            if not selected_file: st.error("Please select an existing SQL file."); st.stop()
            with open(os.path.join(SQL_DIR, selected_file), 'r', encoding='utf-8') as f: final_sql_input = f.read()
            suite_name_prefix = os.path.splitext(selected_file)[0]
        else: # Write Custom SQL Manually
            final_sql_input = st.session_state.auto_custom_sql_input
            if not final_sql_input: st.error("Please enter your custom SQL."); st.stop()
            if SEGMENT_BY_COLUMNS and "{extra_where_conditions}" not in final_sql_input:
                st.error("Segmentation is enabled, but the required '{extra_where_conditions}' placeholder is missing in your custom SQL."); st.stop()
            suite_name_prefix = "custom_sql_suite"

        if not final_sql_input.strip():
            st.error("SQL Query is empty. Please generate, select, or write a query."); st.stop()
        if not auto_spec_input: st.error("Please provide the Data Spec."); st.stop()
        
        suite_name = suite_name_prefix
        
        # Step 2: Test SQL Query before calling Gemini
        with st.spinner("1/4 - Testing SQL query connectivity..."):
            try:
                engine = create_engine(f'trino://{trino_username}:{urllib.parse.quote(trino_password)}@{trino_host}:{trino_port}/', connect_args={'http_scheme': 'https', 'source': 'gx-streamlit-app-sql-test'})
                test_query = final_sql_input.format(dt=db_query_date, extra_where_conditions="") + " LIMIT 1"
                with engine.connect() as conn: conn.execute(text(test_query))
                st.success("✅ SQL query test run successful.")
            except Exception as e:
                st.error(f"SQL query failed to execute: {e}")
                with st.expander("View Failing SQL"):
                    st.code(test_query, language="sql")
                st.stop()
        
        # Step 3: Generate Expectations
        with st.spinner("2/4 - Generating expectation rules..."):
            try:
                with open("prompt_template.txt", "r", encoding="utf-8") as f: prompt_template = f.read()
                prompt = prompt_template.format(auto_spec_input=auto_spec_input)
            except FileNotFoundError: st.error("File not found: 'prompt_template.txt'."); st.stop()
            try:
                model = genai.GenerativeModel('gemini-1.5-flash'); response = model.generate_content(prompt)
                match = re.search(r'\[.*\]', response.text, re.DOTALL)
                if not match: st.error(f"Could not extract JSON from Gemini's response. Response: {response.text}"); st.stop()
                generated_json_str = match.group(0)
                try:
                    parsed_json = json.loads(generated_json_str)
                    st.success("✅ Expectation rules generated.")
                    exp_filepath = os.path.join(EXPECTATIONS_DIR, f"{suite_name}.json")
                    with open(exp_filepath, 'w', encoding='utf-8') as f: json.dump(parsed_json, f, indent=4, ensure_ascii=False)
                    st.success(f"✅ Saved expectation file to '{exp_filepath}'")
                except json.JSONDecodeError as e:
                    st.error(f"Gemini returned invalid JSON: {e}")
                    st.info("You can copy the raw text below, fix it, and use it in Manual Mode.")
                    st.code(generated_json_str, language='json')
                    st.stop()
            except Exception as e: st.error(f"An error occurred during the Gemini API call: {e}"); st.stop()

        with st.expander("View Auto-Generated Artifacts"):
            st.subheader("Final SQL Used")
            st.code(final_sql_input, language="sql")
            st.subheader("Generated Expectation Rules")
            st.json(generated_json_str)
        
        # Step 4: Run full validation
        st.info("3/4 - Starting the full validation process...")
        db_config = (trino_host, trino_port, trino_username, trino_password)
        val_params = (db_query_date, SEGMENT_BY_COLUMNS)
        
        checkpoint_result, data_samples = run_validation_process(final_sql_input, generated_json_str, suite_name, db_config, val_params)
        
        if checkpoint_result:
            st.header("🔬 Data Samples (First 5 Rows)")
            if data_samples:
                for segment_name, sample_df in data_samples.items():
                    with st.expander(f"Data for Segment: {segment_name}"):
                        st.dataframe(sample_df, use_container_width=True)


            st.header("📊 Final Validation Results")
            if checkpoint_result.success: st.success("🎉 **Overall Status: SUCCESS**")
            else: st.error("🚨 **Overall Status: FAILURE**")
            display_validation_errors(checkpoint_result)
            
           
            with st.expander("View Full JSON Result"):
                st.json(checkpoint_result.to_json_dict())

            st.subheader("🔗 Data Docs")
            with st.spinner("Building Data Docs..."):
                context = gx.get_context(); context.build_data_docs()
                docs_sites = context.get_docs_sites_urls()
                if docs_sites:
                    st.info("Copy the path below into a new browser tab to open the full report:")
                    st.code(docs_sites[0]['site_url'], language=None)
                    webbrowser.open(docs_sites[0]['site_url'])
                else:
                    st.warning("Could not generate a path to Data Docs.")