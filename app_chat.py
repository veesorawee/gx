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
import time
import uuid
import webbrowser # เพิ่ม import ที่จำเป็น

# --- 0. App Configuration & Setup ---
st.set_page_config(layout="wide", page_title="Automated Data Validation")

# ตั้งค่าโฟลเดอร์สำหรับเก็บไฟล์
SQL_DIR = "sql"
EXPECTATIONS_DIR = "expectations"
LOGS_DIR = "logs"
# ไม่จำเป็นต้องใช้ STATIC_DIR อีกต่อไป
os.makedirs(SQL_DIR, exist_ok=True)
os.makedirs(EXPECTATIONS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# โหลดค่าจาก .env และตั้งค่า Gemini API
load_dotenv()
GEMINI_AVAILABLE = False
api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
if api_key:
    try:
        genai.configure(api_key=api_key)
        GEMINI_AVAILABLE = True
    except Exception:
        GEMINI_AVAILABLE = False

# --- DB Configuration Check ---
TRINO_HOST = os.getenv('TRINO_HOST')
TRINO_PORT = os.getenv('TRINO_PORT')
TRINO_USERNAME = os.getenv('TRINO_USERNAME')
TRINO_PASSWORD = os.getenv('TRINO_PASSWORD')

if not all([TRINO_HOST, TRINO_PORT, TRINO_USERNAME, TRINO_PASSWORD]):
    st.error("❌ **Database Configuration Not Found!** Please configure TRINO_HOST, TRINO_PORT, TRINO_USERNAME, and TRINO_PASSWORD in your .env file.")
    st.stop()

DB_CONFIG = (TRINO_HOST, TRINO_PORT, TRINO_USERNAME, TRINO_PASSWORD)

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

def parse_select_columns(sql_string: str) -> list[str]:
    try:
        match = re.search(r'SELECT(.*?)FROM', sql_string, re.IGNORECASE | re.DOTALL)
        if not match: return []
        columns_str = match.group(1).replace('\n', ' ').replace('\r', '')
        columns = [col.strip().split(' as ')[-1].strip() for col in columns_str.split(',')]
        return [col for col in columns if col]
    except Exception:
        return []

def generate_automated_sql(screen_name: str, app_type: str, event_type: str, event_category: str, target: str, segment_columns: list) -> str:
    table_name_middle = f"_{app_type}" if app_type == "web" else ""
    table_name = f"jitsu_lm_user{table_name_middle}_{event_type}"
    try:
        with open("sql_template.txt", "r", encoding="utf-8") as f:
            template = f.read()
    except FileNotFoundError:
        st.error("File not found: 'sql_template.txt'. Please create it first.")
        st.stop()
    existing_columns = parse_select_columns(template)
    missing_segment_cols = [col for col in segment_columns if col not in existing_columns]
    if missing_segment_cols:
        cols_to_add = ", ".join(missing_segment_cols) + ", "
        template = re.sub(r'SELECT\s+', f'SELECT {cols_to_add}', template, count=1, flags=re.IGNORECASE)
    where_clauses = [f"dt = '{{dt}}'"]
    if screen_name: where_clauses.append(f"screen_name = '{screen_name}'")
    if event_type in ["click", "impression", "custom"]:
        if event_category: where_clauses.append(f"event_category = '{event_category}'")
    if event_type in ["click", "impression"]:
        if target: where_clauses.append(f"target = '{target}'")
    base_query = template.format(table_name=table_name)
    query = f"{base_query} WHERE {' AND '.join(where_clauses)} {{extra_where_conditions}}"
    return query

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
                    raw_examples = result.result.get("partial_unexpected_list", [])
                    string_examples = [str(item) for item in raw_examples]
                    failed_results_list.append({
                        "Segment": segment, "Expectation": result.expectation_config.expectation_type,
                        "Column": kwargs.get("column", "N/A"), "Expected": str(expected),
                        "Error Count": unexpected_count, "Error %": error_percent,
                        "Failing Examples": string_examples
                    })
    if failed_results_list:
        st.subheader("Failure Summary")
        df = pd.DataFrame(failed_results_list)
        st.data_editor(df, column_config={
                "Error %": st.column_config.ProgressColumn("Error Rate", format="%.2f%%", min_value=0, max_value=100),
                "Failing Examples": st.column_config.ListColumn("Examples", width="large"),
                "Expected": st.column_config.TextColumn("Expected Value", width="medium"),
            }, use_container_width=True, hide_index=True)

def run_validation_process(sql_input, expectations_input, suite_name, db_config, val_params):
    trino_host, trino_port, trino_username, trino_password = db_config
    db_query_date, SEGMENT_BY_COLUMNS = val_params
    context = gx.get_context(project_root_dir=os.getcwd())
    checkpoint_result, data_samples, docs_info = None, None, {"main": None, "segments": {}, "local_path": None}
    process_log = []
    run_name = None
    docs_error = None

    with st.status("🚀 Starting Validation Process...", expanded=True) as status:
        def log_and_write(message):
            process_log.append(message)
            status.write(message)

        run_name = f"run_{suite_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_and_write(f"**Run Name:** `{run_name}` | **Suite:** `{suite_name}`")
        try:
            expectations_list = json.loads(expectations_input)
        except json.JSONDecodeError:
            status.update(label="Input Error", state="error"); st.error("Invalid Expectations JSON format."); return None, None, None, process_log, run_name, None

        log_and_write("🔍 Discovering object columns...")
        schema_expectations = [exp for exp in expectations_list if exp.get('expectation_type') == 'expect_column_values_to_match_json_schema']
        object_columns = sorted(list(set([exp['kwargs']['column'] for exp in schema_expectations])))
        keys_to_flatten_map = {obj_col: [] for obj_col in object_columns}
        if object_columns:
            log_and_write(f"**Identified Parent Object columns:** `{object_columns}`")
            for exp in schema_expectations:
                parent_col = exp['kwargs']['column']
                properties = exp['kwargs'].get('json_schema', {}).get('properties', {})
                keys_to_flatten_map[parent_col].extend(properties.keys())
            all_columns_in_rules = {exp['kwargs']['column'] for exp in expectations_list if 'column' in exp.get('kwargs', {})}
            for col_name in all_columns_in_rules:
                if "_" in col_name:
                    parent = col_name.split('_')[0]
                    if parent in keys_to_flatten_map:
                        key = col_name[len(parent) + 1:]
                        if key not in keys_to_flatten_map[parent]:
                            keys_to_flatten_map[parent].append(key)
            log_and_write("🗺️ **Fields to flatten:**")
            for obj_col, keys in keys_to_flatten_map.items():
                if keys:
                    keys_to_flatten_map[obj_col] = sorted(list(set(keys)))
                    log_and_write(f"&nbsp;&nbsp;&nbsp;↳ For `{obj_col}`: `{keys_to_flatten_map[obj_col]}`")

        log_and_write(f"**1. 🔗 Connecting to DB...**")
        try:
            engine = create_engine(f'trino://{trino_username}:{urllib.parse.quote(trino_password)}@{trino_host}:{trino_port}/', connect_args={'http_scheme': 'https', 'source': 'gx-streamlit-app'})
            with engine.connect() as conn: conn.execute(text("SELECT 1"))
            log_and_write("... ✅ Connection successful.")
        except Exception as e:
            status.update(label="Connection Failed", state="error"); st.error(f"Could not connect to Database: {e}"); return None, None, None, process_log, run_name, None

        segments_to_run = []
        if SEGMENT_BY_COLUMNS:
            log_and_write(f"**2. 🔭 Discovering segments by:** `{', '.join(SEGMENT_BY_COLUMNS)}`")
            try:
                base_query_for_discovery = sql_input.format(dt=db_query_date, extra_where_conditions="")
                discovery_sql = f"SELECT {', '.join(SEGMENT_BY_COLUMNS)} FROM ({base_query_for_discovery}) AS discovery_subquery GROUP BY {', '.join(SEGMENT_BY_COLUMNS)}"
                segments_df = pd.read_sql(text(discovery_sql), engine)
                segments_to_run = segments_df.to_dict('records')
            except Exception as e:
                status.update(label="Error", state="error"); st.error(f"Error discovering segments: {e}"); return None, None, None, process_log, run_name, None
            if not segments_to_run:
                status.update(label="Warning", state="error"); st.warning(f"⚠️ No segments found for date `{db_query_date}`."); return None, None, None, process_log, run_name, None
            log_and_write(f"... ✅ Found **{len(segments_to_run)}** segments.")
        else:
            log_and_write("**2. 🗂️ No segmentation specified.**")
            segments_to_run = [{}]

        context = gx.get_context(); context.add_or_update_expectation_suite(suite_name); validations_to_run = []
        log_and_write("**3. ⚙️ Processing each segment...**")
        data_samples = {}
        for i, segment_filters in enumerate(segments_to_run):
            segment_name = "-".join(str(v) for v in segment_filters.values()).replace(".", "_") if segment_filters else "full_dataset"
            log_and_write(f"&nbsp;&nbsp;&nbsp;↳ **Segment {i+1}/{len(segments_to_run)}:** `{segment_name}`")
            extra_where_conditions = " ".join([f"AND {col} = '{val}'" for col, val in segment_filters.items()])
            final_query = sql_input.format(dt=db_query_date, extra_where_conditions=extra_where_conditions) + " LIMIT 500"
            try:
                df = pd.read_sql(text(final_query), engine)
                if df.empty: log_and_write(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- No data. Skipping."); continue
                for col_name in object_columns:
                    if col_name in df.columns: df[col_name] = df[col_name].apply(lambda x: json.dumps(x) if x is not None else None)
                df = flatten_json_columns(df, keys_to_flatten_map)
                data_samples[segment_name] = df.head(5)
                datasource = context.sources.add_or_update_pandas(f"pandas_datasource_{segment_name}")
                asset_name = f"asset_{segment_name}"
                data_asset = datasource.add_dataframe_asset(name=asset_name)
                batch_request = data_asset.build_batch_request(dataframe=df)
                validator = context.get_validator(batch_request=batch_request, expectation_suite_name=suite_name)
                for exp_config in expectations_list:
                    getattr(validator, exp_config['expectation_type'])(**exp_config["kwargs"])
                validator.save_expectation_suite(discard_failed_expectations=False)
                validations_to_run.append({"batch_request": batch_request, "expectation_suite_name": suite_name})
            except Exception as e:
                tb_str = traceback.format_exc(); log_and_write(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- 🚨 Failed: {e}")
                st.expander(f"Full Error Traceback for Segment: {segment_name}").code(tb_str, language='text')
                continue

        if not validations_to_run:
            status.update(label="Error", state="error"); st.error("No valid batches were created."); return None, None, None, process_log, run_name, None

        log_and_write("**4. 🏁 Running final Great Expectations checkpoint...**")
        checkpoint = context.add_or_update_checkpoint(name=f"checkpoint_{run_name}", run_name_template=run_name,
            action_list=[
                {"name": "store_validation_result", "action": {"class_name": "StoreValidationResultAction"}},
                {"name": "store_evaluation_params", "action": {"class_name": "StoreEvaluationParametersAction"}},
                {"name": "update_data_docs", "action": {"class_name": "UpdateDataDocsAction"}},
            ])
        checkpoint_result = checkpoint.run(validations=validations_to_run)

        log_and_write("**5. 📄 Building Data Docs...**")
        context.build_data_docs()
        
        # --- START: SIMPLIFIED LOCAL PATH APPROACH ---
        log_and_write("... Retrieving local report paths.")
        docs_sites = context.get_docs_sites_urls()
        local_site_url = docs_sites[0]['site_url'] if docs_sites else None
        if local_site_url:
            log_and_write(f"... 📑 Main local report path found: `{local_site_url}`")
            docs_info["local_path"] = local_site_url

            # Get segment-specific paths
            if checkpoint_result:
                for result_identifier in checkpoint_result.run_results.keys():
                    batch_id = result_identifier.batch_identifier
                    match = re.search(r'pandas_datasource_(.*?)-asset_', batch_id)
                    if match:
                        segment_name = match.group(1)
                        segment_url_list = context.get_docs_sites_urls(resource_identifier=result_identifier)
                        if segment_url_list:
                            segment_file_url = segment_url_list[0]['site_url']
                            docs_info["segments"][segment_name] = segment_file_url
                            log_and_write(f"... 📑 Segment report path for '{segment_name}': `{segment_file_url}`")
        else:
            log_and_write("... ⚠️ Could not retrieve local path for Data Docs.")
        # --- END: SIMPLIFIED LOCAL PATH APPROACH ---
                            
        status.update(label="✅ **Validation Process Complete!**", state="complete", expanded=False)

    return checkpoint_result, data_samples, docs_info, process_log, run_name, docs_error

# --- 2. UI Rendering Functions ---

def open_report_in_browser(path):
    """Callback function to open a local file path in the browser."""
    if path:
        webbrowser.open(path)

def display_run_results(run_data):
    """Displays the results of a validation run."""
    st.header(f"📊 Validation Results for: {run_data['name']}")
    st.caption(f"Run ID: `{run_data['id']}` | Timestamp: `{run_data['timestamp']}`")

    docs_info = run_data.get("docs", {})
    local_path = docs_info.get("local_path")
    segment_docs_urls = docs_info.get("segments", {})

    # --- START: NEW LAYOUT FOR DATA DOCS SECTION ---
    st.subheader("📄 Data Docs Reports")
    if local_path:
        # Collect all reports to create buttons
        reports = []
        reports.append({
            "label": "📈 Open Overall Report",
            "path": local_path,
            "type": "primary",
            "key_suffix": "overall"
        })
        for segment_name, report_url in sorted(segment_docs_urls.items()):
            reports.append({
                "label": f"📑 Report for '{segment_name}'",
                "path": report_url,
                "type": "secondary",
                "key_suffix": segment_name
            })
        
        # Create a layout with N columns for buttons to place them side-by-side
        num_buttons = len(reports)
        if num_buttons > 0:
            # Use columns with specific width ratios to keep buttons together
            # The last element in the list acts as a spacer
            col_specs = [1] * num_buttons + [10 - num_buttons] # Adjust spacer size
            cols = st.columns(col_specs)

            for i, report in enumerate(reports):
                with cols[i]:
                    st.button(
                        report["label"],
                        on_click=open_report_in_browser,
                        args=[report["path"]],
                        type=report["type"],
                        key=f"button_{run_data['id']}_{report['key_suffix']}"
                    )
    else:
        st.warning("No report path was generated for this run. Please check the validation process log for errors.")
    st.markdown("---") # Add a separator
    # --- END: NEW LAYOUT ---

    if run_data.get("run_name"):
        st.subheader("🔍 Run Identifier")
        st.text(f"Great Expectations Run Name: {run_data['run_name']}")

    if run_data['status'] == "SUCCESS":
        st.success("🎉 **Overall Status: SUCCESS**")
    else:
        st.error(f"🚨 **Overall Status: {run_data['status']}**")

    if run_data.get("validation_process_log"):
        with st.expander("Show Validation Process Log"):
            for log_entry in run_data["validation_process_log"]:
                st.markdown(log_entry, unsafe_allow_html=True)

    if run_data.get('checkpoint_result'):
        display_validation_errors(run_data['checkpoint_result'])

    if run_data.get('data_samples'):
        st.header("🔬 Data Samples")
        for segment_name, sample_df in run_data['data_samples'].items():
            with st.expander(f"Data for Segment: {segment_name}"):
                st.dataframe(sample_df, use_container_width=True)

    with st.expander("View Full JSON Result (Log)"):
        st.json(run_data['log_data'])

def manual_ui():
    """Renders the original, non-chat UI for manual configuration."""
    st.title("🔩 Manual Data Validation")
    st.info("Configure SQL and Expectation rules manually.")
    st.header("📅 Validation Parameters")
    c1, c2 = st.columns(2)
    yesterday = datetime.date.today() - datetime.timedelta(days=1)
    db_query_date = c1.text_input("Query Date (YYYYMMDD)", value=yesterday.strftime('%Y%m%d'))
    segment_by_columns_str = c2.text_input("Columns to Segment By", help="Comma-separated. Leave empty to validate all data as one batch.")
    SEGMENT_BY_COLUMNS = [col.strip() for col in segment_by_columns_str.split(',') if col.strip()]
    val_params = (db_query_date, SEGMENT_BY_COLUMNS)
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
            st.error("Segmentation is enabled, but the required '{extra_where_conditions}' placeholder is missing in your SQL query."); st.stop()
        suite_name = os.path.splitext(selected_exp_file)[0] if selected_exp_file != "--- Create New ---" else "manual_suite"
        checkpoint_result, data_samples, docs_info, process_log, run_name, docs_error = run_validation_process(manual_sql_input, manual_expectations_input, suite_name, DB_CONFIG, val_params)
        run_id = str(uuid.uuid4())
        run_data = {
            "id": run_id, "name": suite_name, "timestamp": datetime.datetime.now().isoformat(),
            "status": "SUCCESS" if checkpoint_result and checkpoint_result.success else "FAILURE",
            "run_name": run_name,
            "checkpoint_result": checkpoint_result, "data_samples": data_samples, "docs": docs_info,
            "log_data": checkpoint_result.to_json_dict() if checkpoint_result else {"error": "Process failed"},
            "validation_process_log": process_log,
            "docs_error": docs_error
        }
        log_filepath = os.path.join(LOGS_DIR, f"run-{run_id}.json")
        with open(log_filepath, 'w', encoding='utf-8') as f: json.dump(run_data['log_data'], f, indent=4)
        st.session_state.history.insert(0, run_data)
        st.session_state.active_run_id = run_id
        st.rerun()

def chat_ui():
    """Renders the Chat UI for automate mode."""
    st.title("🤖 Automated Data Validation Chat")

    if st.session_state.active_run_id:
        active_run_data = next((run for run in st.session_state.history if run["id"] == st.session_state.active_run_id), None)
        if active_run_data:
            display_run_results(active_run_data)
            st.markdown("---")
            st.info("You can enter a new spec below to start another validation run, or select a new run from the history.")

    if "messages" not in st.session_state: st.session_state.messages = []
    if not st.session_state.messages and not st.session_state.active_run_id:
         st.session_state.messages.append({"role": "assistant", "content": "สวัสดีครับ! ผมคือผู้ช่วยตรวจสอบข้อมูลอัตโนมัติ กรุณาใส่ **Data Spec** ของคุณเพื่อเริ่มต้นได้เลยครับ"})
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.stage == "awaiting_config":
        with st.chat_message("assistant"):
            st.write("เยี่ยมเลย! ตอนนี้กรุณาตั้งค่า SQL และ Validation Parameters ด้านล่างนี้ครับ (ค่าจากการรันครั้งล่าสุดได้ถูกใส่ไว้ให้แล้ว)")
            with st.form("automate_config_form"):
                st.subheader("Configure SQL & Validation Parameters")
                
                yesterday = datetime.date.today() - datetime.timedelta(days=1)
                prev_val_params = st.session_state.get("val_params", {})
                prev_sql_config = st.session_state.get("sql_config", {})

                db_query_date = st.text_input("Query Date (YYYYMMDD)", value=prev_val_params.get('date', yesterday.strftime('%Y%m%d')))
                segment_by_columns_str = st.text_input("Columns to Segment By", value=", ".join(prev_val_params.get('segments', [])), help="Comma-separated. Leave empty for one batch.")
                
                sql_source_options = ["Generate from Target Inputs", "Select Existing SQL File", "Write Custom SQL Manually"]
                sql_source_index = sql_source_options.index(prev_sql_config.get("source", "Generate from Target Inputs"))
                sql_source = st.radio("SQL Source", sql_source_options, index=sql_source_index, key="auto_sql_source_selector", horizontal=True)

                if sql_source == "Generate from Target Inputs":
                    c1, c2 = st.columns(2)
                    app_type_options = ["native", "app", "web"]
                    app_type_index = app_type_options.index(prev_sql_config.get("app_type", "native"))
                    auto_app_type = c1.selectbox("App Type", app_type_options, index=app_type_index, key="auto_app_type_gen")
                    
                    event_type_options = ["click", "impression", "pageview", "custom"]
                    event_type_index = event_type_options.index(prev_sql_config.get("event_type", "click"))
                    auto_event_type = c2.selectbox("Event Type", event_type_options, index=event_type_index, key="auto_event_type_gen")
                    
                    c3, c4, c5 = st.columns(3)
                    auto_screen_name = c3.text_input("Screen Name", value=prev_sql_config.get("screen_name", ""), key="auto_screen_name_gen")
                    auto_event_category = c4.text_input("Event Category", value=prev_sql_config.get("event_category", ""), key="auto_event_category_gen")
                    auto_target = c5.text_input("Target", value=prev_sql_config.get("target", ""), key="auto_target_gen")

                elif sql_source == "Select Existing SQL File":
                    sql_files = [""] + list_files(SQL_DIR, "sql")
                    sql_file_index = sql_files.index(prev_sql_config.get("file", "")) if prev_sql_config.get("file") in sql_files else 0
                    selected_manual_sql = st.selectbox("Select from existing SQL files:", sql_files, index=sql_file_index, key="auto_sql_file_selector")

                else: 
                    custom_sql_input = st.text_area("SQL Query Input:", value=prev_sql_config.get("custom_sql", ""), key="auto_custom_sql_input", height=150)
                
                submitted = st.form_submit_button("🚀 Run Validation Process")
                if submitted:
                    st.session_state.val_params = {"date": db_query_date, "segments": [col.strip() for col in segment_by_columns_str.split(',') if col.strip()]}
                    st.session_state.sql_config = {"source": sql_source}
                    if sql_source == "Generate from Target Inputs": st.session_state.sql_config.update({"app_type": auto_app_type, "event_type": auto_event_type, "screen_name": auto_screen_name, "event_category": auto_event_category, "target": auto_target})
                    elif sql_source == "Select Existing SQL File": st.session_state.sql_config["file"] = selected_manual_sql
                    else: st.session_state.sql_config["custom_sql"] = custom_sql_input
                    st.session_state.stage = "processing"
                    st.rerun()

    if prompt := st.chat_input("Enter a new data spec to start..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.spec = prompt
        st.session_state.stage = "awaiting_config"
        st.rerun()

    if st.session_state.stage == "processing":
        with st.chat_message("assistant"):
            val_params_tuple = (st.session_state.val_params["date"], st.session_state.val_params["segments"])
            final_sql, final_expectations, suite_name = "", "", "default_suite"
            
            with st.spinner("1/3 - Preparing SQL Query..."):
                sql_cfg = st.session_state.sql_config
                if sql_cfg["source"] == "Generate from Target Inputs":
                    final_sql = generate_automated_sql(sql_cfg["screen_name"], sql_cfg["app_type"], sql_cfg["event_type"], sql_cfg["event_category"], sql_cfg["target"], val_params_tuple[1])
                    suite_name = f"{sql_cfg['screen_name']}_{sql_cfg['event_category']}_{sql_cfg['target']}"
                elif sql_cfg["source"] == "Select Existing SQL File":
                    with open(os.path.join(SQL_DIR, sql_cfg["file"]), 'r', encoding='utf-8') as f: final_sql = f.read()
                    suite_name = os.path.splitext(sql_cfg["file"])[0]
                else: final_sql = sql_cfg["custom_sql"]; suite_name = "custom_sql_suite"
                st.success("SQL Query prepared.")
                st.code(final_sql, language="sql")

            with st.spinner("2/3 - Generating expectation rules with Gemini..."):
                if not GEMINI_AVAILABLE: st.error("Gemini API not available."); st.stop()
                try:
                    with open("prompt_template.txt", "r", encoding="utf-8") as f: prompt_template = f.read()
                    prompt = prompt_template.format(auto_spec_input=st.session_state.spec)
                    model = genai.GenerativeModel('gemini-1.5-flash'); response = model.generate_content(prompt)
                    match = re.search(r'\[.*\]', response.text, re.DOTALL)
                    if match:
                        final_expectations = match.group(0); json.loads(final_expectations)
                        st.success("Expectation rules generated.")
                        st.json(final_expectations)
                    else: 
                        st.error("Failed to generate valid JSON rules from Gemini."); st.stop()
                except Exception as e: 
                    st.error(f"Error during Gemini call: {e}"); st.stop()

            st.info("3/3 - Running full validation process...")
            checkpoint_result, data_samples, docs_info, process_log, run_name, docs_error = run_validation_process(final_sql, final_expectations, suite_name, DB_CONFIG, val_params_tuple)
            
            run_id = str(uuid.uuid4())
            run_data = {
                "id": run_id, "name": suite_name, "timestamp": datetime.datetime.now().isoformat(),
                "status": "SUCCESS" if checkpoint_result and checkpoint_result.success else "FAILURE",
                "run_name": run_name,
                "checkpoint_result": checkpoint_result, "data_samples": data_samples, "docs": docs_info,
                "log_data": checkpoint_result.to_json_dict() if checkpoint_result else {"error": "Process failed"},
                "chat_log": st.session_state.messages.copy(),
                "validation_process_log": process_log,
                "docs_error": docs_error
            }
            log_filepath = os.path.join(LOGS_DIR, f"run-{run_id}.json")
            with open(log_filepath, 'w', encoding='utf-8') as f: json.dump(run_data['log_data'], f, indent=4)
            
            st.session_state.history.insert(0, run_data)
            st.session_state.active_run_id = run_id
            st.session_state.stage = "awaiting_spec"
            st.rerun()

# --- 3. Main App Logic & State Initialization ---
if "history" not in st.session_state: st.session_state.history = []
if "active_run_id" not in st.session_state: st.session_state.active_run_id = None
if "stage" not in st.session_state: st.session_state.stage = "awaiting_spec"
if 'mode' not in st.session_state: st.session_state.mode = 'Automate'
if "val_params" not in st.session_state: st.session_state.val_params = {}
if "sql_config" not in st.session_state: st.session_state.sql_config = {}


# --- Sidebar ---
with st.sidebar:
    st.header("Controls")
    if st.button("➕ New Run", use_container_width=True):
        st.session_state.active_run_id = None
        st.session_state.stage = "awaiting_spec"
        st.session_state.messages = []
        st.session_state.val_params = {}
        st.session_state.sql_config = {}
        if 'spec' in st.session_state:
            del st.session_state['spec']
        st.rerun()

    st.radio("Select Mode:", ["Automate", "Manual"], key="mode", on_change=lambda: st.session_state.update(active_run_id=None, messages=[], stage="awaiting_spec"))
    st.markdown("---")
    st.header("History")
    if not st.session_state.history:
        st.caption("No runs yet.")
    else:
        for run in st.session_state.history:
            icon = "✅" if run['status'] == "SUCCESS" else "🚨"
            if st.button(f"{icon} {run['name']} ({run['timestamp'].split('T')[0]})", key=run['id'], use_container_width=True):
                st.session_state.active_run_id = run['id']
                st.session_state.stage = "awaiting_spec" 
                st.session_state.messages = []
                st.rerun()

# --- Main Panel Rendering ---
if st.session_state.mode == "Automate":
    chat_ui()
else: # Manual Mode
    if st.session_state.active_run_id:
        active_run_data = next((run for run in st.session_state.history if run["id"] == st.session_state.active_run_id), None)
        if active_run_data:
            display_run_results(active_run_data)
    else:
        manual_ui()