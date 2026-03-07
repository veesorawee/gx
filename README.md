# Data Validation Pipeline with Great Expectations (v6)

A powerful Streamlit-based data validation tool that leverages Great Expectations and Google Gemini AI to ensure your data quality! 🚀

[คลิกที่นี่เพื่อข้ามไปเวอร์ชันภาษาไทย (Skip to Thai Version)](#thai-version)

---

## 🌟 Key Features

- 🔄 **Dual Validation Modes**: 
  - **Manual Mode**: Full control over SQL and expectations.
  - **Automate Mode**: Describe your validation needs in plain English and let AI (Gemini) generate SQL and rules.
- 🗄️ **Trino Integration**: Direct connection to your Trino database for real-time validation.
- 📊 **Dynamic Segmentation**: Automatically discover and validate data across multiple dimensions (e.g., by screen, event, or specific segments).
- 🛑 **Waive & Refine**: Exclude specific failed rules (false positives) directly from the UI and instantly regenerate your metrics and reports.
- 📄 **Professional PDF Reporting**: Automated generation of "Tracking Validation" PDF summaries for sharing results.
- 📚 **Great Expectations Data Docs**: Deep-dive into detailed HTML reports for every validation run.
- 🤖 **AI-Powered**: Uses Google Gemini to translate business requirements into technical validation rules.

---

## 🛠️ Step-by-Step Beginner Setup Guide (macOS)

If you've never used Python or run a coding project before, don't worry! Just follow these steps exactly.

### Step 1: Install "Homebrew" (A tool to install other tools)
Open the **Terminal** app on your Mac (you can find it in Applications > Utilities, or search for it using Spotlight). Copy the following line, paste it into the Terminal, and press Enter:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
*(If it asks for your Mac password, type it. You won't see the letters appear, but it is typing!)*

### Step 2: Install System Dependencies
This app creates PDF reports, which needs a few special system files. Paste this in Terminal and press Enter:
```bash
brew install pango libffi glib
```

### Step 3: Download the Project
Now, let's download the code to your computer. In the same Terminal, run these two commands one by one:
```bash
git clone https://github.com/veesorawee/gx.git
cd gx
```

### Step 4: Create a "Virtual Environment"
A virtual environment is like a sandbox so this app doesn't mess with other things on your Mac. Run these two commands:
```bash
python3 -m venv venv
source venv/bin/activate
```
*(You should now see `(venv)` at the beginning of your Terminal line. This means the sandbox is active!)*

### Step 5: Install Python Packages
Now we install the specific Python libraries this app needs:
```bash
pip install -r requirements.txt
```
*(This might take a minute or two. Wait until it finishes completely.)*

### Step 6: Setup Passwords (Configuration)
The app needs to connect to a database and use AI. We need to give it the passwords securely.
First, copy the example settings file:
```bash
cp .env.example .env
```
Now, you need to edit this `.env` file. You can open it in a text editor (like TextEdit or VS Code). Make sure you fill in:
- `TRINO_HOST`, `TRINO_PORT`, `TRINO_USERNAME`, `TRINO_PASSWORD` (Your database details)
- `GEMINI_API_KEY` (Get this for free from [Google AI Studio](https://aistudio.google.com/))

---

## 🚀 Running the App

Every time you want to use the app, open Terminal, and run these commands to start it:

```bash
# 1. Go to the app folder (if you aren't already there)
cd ~/gx

# 2. Activate the virtual environment
source venv/bin/activate

# 3. Start the app!
streamlit run app_v6.py
```
Your web browser will automatically open to `http://localhost:8501`.

---

## 🚨 Common Errors & Troubleshooting

Here are common issues and exactly how to fix them:

**Error: `command not found: streamlit` or `ModuleNotFoundError: No module named '...'`**
- **Why it happens:** You forgot to activate your "sandbox" (virtual environment).
- **The Fix:** Run `source venv/bin/activate` in your Terminal before running the app.

**App stuck on loading / Error: `Database Configuration Not Found!`**
- **Why it happens:** The app cannot find your `.env` file, or you left the passwords blank.
- **The Fix:** Make sure your file is named EXACTLY `.env` (not `.env.txt` or `.env.example`). Open it and check that `TRINO_HOST` and passwords are filled in.

**Error: `Failed to load WeasyPrint...` or PDF Generation Fails**
- **Why it happens:** Your Mac is missing the libraries from Step 2.
- **The Fix:** Close the app (Press `Ctrl + C` in Terminal) and run: `brew install pango libffi glib`.

**Error: Port `8501` is already in use**
- **Why it happens:** You already have the app running in another Terminal window.
- **The Fix:** Find the other Terminal window and stop it by pressing `Ctrl + C`, OR close Terminal completely and restart.

---

<a name="thai-version"></a>
# ระบบตรวจสอบคุณภาพข้อมูลด้วย Great Expectations (v6) [ภาษาไทย]

เครื่องมือตรวจสอบคุณภาพข้อมูล (Data Validation) บน Streamlit ที่ผสานพลังของ Great Expectations และ Google Gemini AI เพื่อความมั่นใจในคุณภาพข้อมูลของคุณ! 🚀

---

## 🌟 คุณสมบัติเด่น

- 🔄 **โหมดการทำงาน 2 แบบ**: 
  - **Manual Mode**: เขียน SQL และกำหนดกฎการตรวจสอบ (Expectations) ด้วยตัวเอง
  - **Automate Mode**: อธิบายกฎการตรวจสอบด้วยภาษาไทย/อังกฤษทั่วไป แล้วให้ AI (Gemini) สร้าง SQL และกฎให้โดยอัตโนมัติ
- 🗄️ **เชื่อมต่อ Trino**: ต่อตรงกับฐานข้อมูล Trino เพื่อรันการตรวจสอบได้ทันที
- 📊 **แยกกลุ่มข้อมูลอัตโนมัติ (Segmentation)**: ค้นหาและตรวจสอบข้อมูลแยกตามมิติต่างๆ (เช่น แยกตามหน้าจอ, ประเภทกิจกรรม) ได้โดยอัตโนมัติ
- 🛑 **ระบบ Waive กฎ**: สามารถเลือกข้ามกฎบางข้อที่ไม่ผ่าน (False Positives) ได้จากหน้าจอ และคำนวณคะแนนใหม่พร้อมออกรีพอร์ตทันที
- 📄 **รายงานสรุป PDF**: สร้างไฟล์ PDF "Tracking Validation" ระดับมืออาชีพเพื่อส่งต่อผลลัพธ์ได้ง่ายๆ
- 📚 **Data Docs**: เข้าถึงรายงาน HTML รายละเอียดสูงจาก Great Expectations ได้ทุกลำดับการรัน
- 🤖 **พลัง AI**: ใช้ Google Gemini ในการเปลี่ยนความต้องการทางธุรกิจให้เป็นกฎทางเทคนิคที่ใช้งานได้จริง

---

## 🛠️ คู่มือติดตั้งฉบับจับมือทำทีละขั้นตอน (สำหรับผู้เริ่มต้นระบบ macOS)

ถ้าคุณไม่เคยเขียนโปรแกรมหรือรันโค้ดมาก่อน ไม่ต้องกังวล! ทำตามขั้นตอนนี้ได้เลย

### ขั้นที่ 1: ติดตั้ง "Homebrew" (เครื่องมือช่วยติดตั้งโปรแกรม)
เปิดแอป **Terminal** ในเครื่อง Mac (ค้นหาโปรแกรมในแอปพลิเคชัน หรือใช้คำค้นหา Spotlight) แล้วคัดลอกคำสั่งนี้ไปวาง กด Enter:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
*(ถ้าระบบถามรหัสผ่านเครื่อง Mac ให้พิมพ์รหัสของคุณแล้วกด Enter ตอนพิมพ์จะไม่เห็นตัวอักษรขยับ ไม่ต้องแปลกใจ มันกำลังพิมพ์อยู่!)*

### ขั้นที่ 2: ติดตั้งไฟล์ระบบที่จำเป็น
แอปนี้มีการสร้างรายงาน PDF ซึ่งต้องใช้ไฟล์ระบบพิเศษ นำคำสั่งนี้ไปวางใน Terminal แล้วกด Enter:
```bash
brew install pango libffi glib
```

### ขั้นที่ 3: ดาวน์โหลดโปรเจกต์
ใน Terminal อันเดิม รันคำสั่งสองบรรทัดนี้ทีละบรรทัด:
```bash
git clone https://github.com/veesorawee/gx.git
cd gx
```

### ขั้นที่ 4: สร้างกระบะทรายจำลองสำหรับติดตั้ง (Virtual Environment)
เพื่อไม่ให้แอปนี้ไปกวนระบบของเครื่อง เราจะสร้างสภาพแวดล้อมจำลอง พิมพ์สองคำสั่งนี้:
```bash
python3 -m venv venv
source venv/bin/activate
```
*(สังเกตว่าตอนนี้จะมีคำว่า `(venv)` ขึ้นมาด้านหน้าสุดของบรรทัดใน Terminal แปลว่าคุณอยู่ในสภาพแวดล้อมจำลองแล้ว!)*

### ขั้นที่ 5: ติดตั้งโปรแกรมย่อย (Python Packages)
สั่งให้ระบบดาวน์โหลดเครื่องมือที่แอปนี้ต้องการ:
```bash
pip install -r requirements.txt
```
*(ขั้นตอนนี้อาจใช้เวลา 1-2 นาที รอจนกว่าจะโหลดเสร็จสมบูรณ์)*

### ขั้นที่ 6: ใส่รหัสผ่านและคีย์ (Configuration)
แอปต้องเชื่อมต่อกับ Database และใช้ปัญญาประดิษฐ์ (AI) เราต้องนำรหัสไปใส่ไว้ในไฟล์
ขั้นแรก คัดลอกไฟล์ตัวอย่างด้วยคำสั่งนี้:
```bash
cp .env.example .env
```
จากนั้น ให้คุณเปิดไฟล์ที่ชื่อว่า `.env` ขึ้นมา (เปิดด้วยโปรแกรมอะไรก็ได้ เช่น TextEdit) แล้วเติมข้อมูลให้ครบ:
- `TRINO_HOST`, `TRINO_PORT`, `TRINO_USERNAME`, `TRINO_PASSWORD` (ข้อมูลฐานข้อมูลของคุณ)
- `GEMINI_API_KEY` (ขอได้ฟรีที่ [Google AI Studio](https://aistudio.google.com/))

---

## 🚀 วิธีการรันแอปพลิเคชัน

ทุกครั้งที่คุณต้องการใช้งานแอป ให้เปิด Terminal ขึ้นมา แล้วรันคำสั่ง 3 บรรทัดนี้:

```bash
# 1. เข้าไปในโฟลเดอร์ของแอป (ถ้าไม่ได้อยู่ในโฟลเดอร์นี้อยู่แล้ว)
cd ~/gx

# 2. เปิดใช้งานสภาพแวดล้อมจำลอง
source venv/bin/activate

# 3. รันแอปพลิเคชัน!
streamlit run app_v6.py
```
เบราว์เซอร์จะเปิดขึ้นมาที่หน้า `http://localhost:8501` โดยอัตโนมัติ

---

## 🚨 ปัญหาที่พบบ่อยและวิธีไข (Troubleshooting)

รวมรวบปัญหาที่คุณอาจจะเจอ และวิธีแก้แบบเป๊ะๆ:

**แจ้งเตือน: `command not found: streamlit` หรือ `ModuleNotFoundError: No module named '...'`**
- **สาเหตุ:** คุณลืมเปิดใช้งาน "กระบะทราย" (Virtual Environment) ในขั้นที่ 4
- **วิธีแก้:** พิมพ์คำสั่ง `source venv/bin/activate` ใน Terminal ก่อนที่จะรันแอปทุกครั้ง

**แอปค้างหน้าโหลด / แจ้งเตือน: `Database Configuration Not Found!`**
- **สาเหตุ:** แอปหาไฟล์ `.env` ไม่เจอ หรือคุณไม่ได้ใส่รหัสผ่านในไฟล์
- **วิธีแก้:** ตรวจสอบให้แน่ใจว่าไฟล์ชื่อ `.env` พอดีเป๊ะ (ไม่ใช่ `.env.txt`) ให้เปิดไฟล์ขึ้นมาเช็คว่าใส่ข้อมูล TRINO และ GEMINI ครบถ้วนแล้ว

**แจ้งเตือน: `Failed to load WeasyPrint...` หรือระบบปริ้น PDF พัง**
- **สาเหตุ:** เครื่อง Mac ของคุณอาจยังไม่ได้ลงโปรแกรมในขั้นที่ 2
- **วิธีแก้:** ปิดแอปก่อน (กด `Ctrl + C` ใน Terminal) แล้วรันคำสั่ง: `brew install pango libffi glib`

**แจ้งเตือน: Port `8501` is already in use**
- **สาเหตุ:** คุณเปิดแอปค้างไว้อยู่แล้วในหน้าต่าง Terminal อื่น
- **วิธีแก้:** หาหน้าต่าง Terminal นั้นให้เจอแล้วกด `Ctrl + C` เพื่อปิดแอปเก่า หรือจะปิดโปรแกรม Terminal ทิ้งทั้งหมดแล้วเปิดขึ้นมาใหม่เลยก็ได้

---

## 📁 โครงสร้างโปรเจกต์

```text
gx/
├── app_v6.py           # ไฟล์หลักของโปรแกรม (Main Entry Point)
├── .env                # ไฟล์เก็บรหัสผ่านและ API Key (ห้ามอัปโหลด!)
├── requirements.txt    # รายการแพ็กเกจ Python ที่ต้องใช้
├── sql/                # โฟลเดอร์เก็บไฟล์ SQL
├── expectations/       # โฟลเดอร์เก็บกฎการตรวจสอบแบบ JSON
├── reports/            # โฟลเดอร์สำหรับเก็บรายงาน PDF ที่สร้างขึ้น
└── logs/               # เก็บประวัติการทำงานและข้อผิดพลาดของ AI
```