import os
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
from langchain_community.document_loaders import PyPDFLoader
import google.generativeai as genai
from dotenv import load_dotenv
import pandas as pd
import json
import time

# ==============================================================================
# --- ส่วนที่ 1: การตั้งค่าเริ่มต้น (Initial Setup) ---
# ==============================================================================

# 1. โหลด Environment Variables (สำหรับ Gemini API Key)
load_dotenv()

# 2. ตั้งค่า Gemini API
try:
    # อ่านจาก os.environ แทน
    api_key = os.environ.get("GOOGLE_API_KEY") 
    if not api_key:
        raise ValueError("ไม่พบ GOOGLE_API_KEY ใน Secrets")
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"🚨 ไม่สามารถตั้งค่า Gemini API ได้: {e}")
    st.stop()

# 3. ตั้งค่า Firebase Admin SDK
import base64
try:
    if not firebase_admin._apps:
        # อ่าน Base64 string จาก Secrets
        firebase_secret_b64 = os.environ.get("FIREBASE_SERVICE_ACCOUNT_BASE64")
        if not firebase_secret_b64:
            raise ValueError("ไม่พบ FIREBASE_SERVICE_ACCOUNT_BASE64 ใน Secrets")
        
        # ถอดรหัสกลับมาเป็น dictionary
        decoded_secret = base64.b64decode(firebase_secret_b64)
        service_account_info = json.loads(decoded_secret)
        
        cred = credentials.Certificate(service_account_info)
        firebase_admin.initialize_app(cred)
    db = firestore.client()
except Exception as e:
    st.error(f"🚨 ไม่สามารถเชื่อมต่อ Firebase ได้: {e}")
    st.stop()

# ==============================================================================
# --- ส่วนที่ 2: ฟังก์ชันการทำงาน (Functions) ---
# ==============================================================================

def extract_metadata_with_function_calling(text_data: str) -> dict:
    """
    ใช้ Gemini และ Function Calling เพื่อสกัด Metadata ที่สำคัญจากเนื้อหา PDF
    """
    # กำหนด Tool สำหรับสกัด Metadata
    metadata_tool = genai.protos.Tool(
        function_declarations=[
            genai.protos.FunctionDeclaration(
                name='log_motor_metadata',
                description="บันทึกข้อมูล metadata ที่สำคัญของการทดสอบมอเตอร์",
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        'equipment_no': genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="หมายเลข Equipment: ของมอเตอร์ เช่น '723P1201_1'"
                        ),
                        'test_timestamp': genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="วันและเวลาที่ทำการทดสอบ ในรูปแบบ YYYYMMDD-HH:MM:SS เช่น '20250616-10:29:43'"
                        ),
                        'test_location': genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="สถานที่หรือวิธีการทดสอบ เช่น 'Direct Test at Motor' หรือ 'At MCC'"
                        ),
                    },
                    required=['equipment_no', 'test_timestamp', 'test_location']
                )
            )
        ]
    )

    model = genai.GenerativeModel(
        model_name='gemini-1.5-flash-latest',
        tools=[metadata_tool]
    )
    
    prompt = f"""
    จากข้อความรายงานผลการทดสอบมอเตอร์ที่ให้มานี้ ให้คุณสกัดข้อมูล metadata ที่สำคัญ 3 อย่างออกมา:
    1. หมายเลข Equipment (มองหาคำว่า 'Equipment:') มักจะอยู่ด้านบนของ file
    2. วันและเวลาที่ทดสอบ (มักจะมีรูปแบบเป็น YYYYMMDD-HH:MM:SS) มักจะอยู่ด้านบน ของผลการวัด MCA เช่นด้านบนของค่า Resistance
    3. สถานที่/วิธีการทดสอบ (มองหาคำว่า 'Direct Test at Motor') มักจะอยู่ด้านล่างของคำว่า Frequency แต่ถ้าไม่เจอคำว่า Direct Test at Motor จริงๆ ให้ใช้ 'At MCC'

    เมื่อพบข้อมูลทั้งหมดแล้ว ให้เรียกใช้เครื่องมือ 'log_motor_metadata'

    ข้อความ:
    ---
    {text_data}
    ---
    """
    
    response = model.generate_content(prompt)
    
    try:
        tool_call = response.candidates[0].content.parts[0].function_call
        if tool_call.name == 'log_motor_metadata':
            return tool_call.args
    except (IndexError, AttributeError, KeyError) as e:
        raise ValueError(f"ไม่สามารถสกัด Metadata ได้: {e}")

    raise ValueError("ไม่พบ Metadata ที่ต้องการในไฟล์นี้")

def get_analysis_from_data(input_data_text, manual_context, user_query):
    """
    ฟังก์ชันกลางสำหรับวิเคราะห์ผล โดยสั่งให้ Gemini ตอบกลับเป็น JSON พร้อมคำอธิบาย
    """
    template = """
    คุณคือ "AIMCA" ผู้เชี่ยวชาญด้านการวิเคราะห์ผล Motor Circuit Analyzer ที่มีความแม่นยำสูง
    หน้าที่ของคุณคือวิเคราะห์ "ข้อมูลผลการตรวจวัด" โดยอ้างอิงจาก "บริบทจากคู่มือ"
    และ **ต้อง** ตอบกลับเป็นรูปแบบ JSON ที่กำหนดไว้เท่านั้น

    ---
    **[บริบทจากคู่มือการใช้งาน (Context)]**
    {manual_context}
    ---
    **[ข้อมูลผลการตรวจวัด (Input Data)] โดยมีค่า Resistance (Ohm), Impedance (Ohm), Inductance (mH), Phase Angle (°) มี column สุดท้ายเป็น %unbalance**
    {input_data}
    ---
    **[คำสั่ง/คำถามเพิ่มเติมจากผู้ใช้ (User Query)]**
    {user_query}
    ---

    **[รูปแบบ JSON ที่ต้องตอบกลับ (JSON Output Format)]**
    โปรดตอบกลับเป็น JSON object ที่มีโครงสร้างดังนี้เท่านั้น อย่าใส่ข้อความอื่นนอกเหนือจาก JSON:
    {{
      "status": "สถานะโดยรวมของมอเตอร์ โดยเลือกคำใดคำหนึ่งจาก: [GOOD, WARNING, CRITICAL]",
      "overall_summary": "สรุปผลโดยรวมสั้นๆ เป็นประโยค (เช่น 'มอเตอร์อยู่ในสภาพปกติ', 'ควรเฝ้าระวัง', 'มีความผิดปกติ')",
      "detailed_analysis": "คำวิเคราะห์เชิงลึกแต่ละรายการ แสดงเป็น bullet point โดยอธิบายแต่ละรายการเทียบกับเกณฑ์ และอธิบายว่าค่าเหล่านั้นส่งผลมาจากอะไรบ้างหรือเป็นปัญหาจากอะไรได้บ้าง",
      "recommendations": "คำแนะนำที่ปฏิบัติได้จริงเป็นข้อความ",
      "graph_data": [
        {{
          "parameter": "ชื่อพารามิเตอร์ที่สำคัญ เช่น 'Phase Imbalance (%)'",
          "measured_value": ค่าที่วัดได้ (ตัวเลขเท่านั้น),
          "standard_value_min": ค่ามาตรฐานต่ำสุด (ตัวเลข, ถ้าไม่มีให้ใส่ null),
          "standard_value_max": ค่ามาตรฐานสูงสุด (ตัวเลข, ถ้าไม่มีให้ใส่ null)
        }},
        {{
          "parameter": "Resistance Imbalance (%)",
          "measured_value": 1.5,
          "standard_value_min": null,
          "standard_value_max": 5.0
        }}
      ]
    }}
    ตัวอย่าง: หากค่า Phase Imbalance วัดได้ 2.1% และเกณฑ์คือไม่เกิน 3% ให้ใส่ "measured_value": 2.1, "standard_value_max": 3.0
    """
    prompt = template.format(
        manual_context=manual_context,
        input_data=input_data_text,
        user_query=user_query if user_query else "โปรดวิเคราะห์ผลการตรวจวัดทั้งหมดโดยละเอียด และให้คำแนะนำเบื้องต้น"
    )
    
    # ตั้งค่าให้ Gemini ตอบกลับเป็น JSON
    generation_config = genai.types.GenerationConfig(
        response_mime_type="application/json"
    )
    
    model = genai.GenerativeModel('gemini-2.0-flash') #gemini-1.5-flash-latest
    response = model.generate_content(prompt, generation_config=generation_config)
    
    # แปลงผลลัพธ์ JSON string เป็น Python dictionary
    return json.loads(response.text)

def migrate_historic_data(root_folder: str, site_name: str):
    """
    วนลูปผ่านโฟลเดอร์เพื่อประมวลผลไฟล์ PDF และบันทึกลง Firebase
    """
    print(f"🚀 เริ่มการย้ายข้อมูลสำหรับไซต์: {site_name}")
    
    for filename in os.listdir(root_folder):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(root_folder, filename)
            print(f"\n📄 กำลังประมวลผลไฟล์: {filename}")
            
            try:
                # 1. สกัดข้อความดิบ
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                raw_text = "\n\n".join([doc.page_content for doc in documents])

                # 2. สกัด Metadata
                print("   - กำลังสกัด Metadata...")
                metadata = extract_metadata_with_function_calling(raw_text)
                equipment_no = metadata.get('equipment_no')
                # แปลง timestamp ให้เป็นรูปแบบที่จัดเรียงได้ง่าย
                test_date = pd.to_datetime(metadata.get('test_timestamp'), format='%Y%m%d-%H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S')

                # 3. วิเคราะห์ข้อมูลเชิงลึก
                print("   - กำลังวิเคราะห์ข้อมูล MCA...")
                # (สมมติว่าคุณมี vectorstore ของคู่มือพร้อมใช้งาน)
                # relevant_docs = retriever.invoke(...) 
                # manual_context = "..."
                # ตอนเริ่มต้น อาจจะยังไม่ต้องใช้ context จากคู่มือก็ได้
                manual_context_placeholder = "ใช้เกณฑ์การประเมินทั่วไปสำหรับมอเตอร์ AC"
                analysis_result = get_analysis_from_data(raw_text, manual_context_placeholder, "")

                # 4. บันทึกลง Firebase
                print("   - กำลังบันทึกผลลง Firebase...")
                doc_ref = db.collection('sites').document(site_name) \
                          .collection('motors').document(equipment_no) \
                          .collection('analyses').document(test_date)
                
                doc_data = {
                    'timestamp': pd.to_datetime(metadata.get('test_timestamp'), format='%Y%m%d-%H:%M:%S'),
                    'analysis_data': analysis_result,
                    'test_location': metadata.get('test_location'),
                    'original_filename': filename
                }
                doc_ref.set(doc_data)
                
                print(f"   ✅ บันทึกข้อมูลสำหรับ {equipment_no} วันที่ {test_date} สำเร็จ!")
                print("   ...หน่วงเวลา 15 วินาทีเพื่อป้องกัน Rate Limit...")
                time.sleep(15)

            except Exception as e:
                print(f"   ❌ เกิดข้อผิดพลาดกับไฟล์ {filename}: {e}")

# ==============================================================================
# --- ส่วนที่ 3: ส่วนการรันโปรแกรมหลัก (Main Execution) ---
# ==============================================================================

if __name__ == "__main__":
    # --- กำหนดค่าและรัน ---
    # จัดระเบียบไฟล์ของคุณตามโครงสร้างนี้
    # historic_data/
    # ├── SITE_A/
    # │   ├── report1.pdf
    # │   └── report2.pdf
    # └── SITE_B/
    #     └── report3.pdf
    
    historic_root = "historic_data"
    sites_to_migrate = os.listdir(historic_root)

    for site in sites_to_migrate:
        site_folder_path = os.path.join(historic_root, site)
        if os.path.isdir(site_folder_path):
            migrate_historic_data(site_folder_path, site)

    print("\n🎉 การย้ายข้อมูลทั้งหมดเสร็จสิ้น!")