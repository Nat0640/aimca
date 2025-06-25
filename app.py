# app.py - AIMCA (เวอร์ชันรองรับการวิเคราะห์จาก CSV และ PDF)

# --- 1. Import ไลบรารีที่จำเป็น ---
import streamlit as st
import pandas as pd
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from io import StringIO
import os
import tempfile
import json

# --- 2. ตั้งค่าเริ่มต้นและโหลดโมเดล ---
try:
    # อ่านจาก os.environ แทน
    api_key = os.environ.get("GOOGLE_API_KEY") 
    if not api_key:
        raise ValueError("ไม่พบ GOOGLE_API_KEY ใน Secrets")
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"🚨 ไม่สามารถตั้งค่า Gemini API ได้: {e}")
    st.stop()

VECTOR_DB_PATH = "vectorstore/db_chroma"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

@st.cache_resource
def load_components():
    """โหลดส่วนประกอบที่ใช้เวลานาน เช่น Embedding Model และ Vector DB หลัก"""
    print("กำลังโหลดส่วนประกอบหลัก...")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cpu'})
    vectordb = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embedding_model)
    print("โหลดส่วนประกอบทั้งหมดเรียบร้อยแล้ว!")
    return vectordb.as_retriever(search_kwargs={'k': 5})

def clear_analysis_state():
    """Callback function to clear all analysis-related session state."""
    st.session_state.analysis_result = None
    st.session_state.chat_history = []
    st.session_state.manual_context = None
    st.session_state.extracted_pdf_text = None
    st.session_state.dataframe_for_graph = None # เพิ่ม state ใหม่สำหรับเก็บ DataFrame

# --- 3. ฟังก์ชันหลักในการทำงาน ---
# --- สร้าง Tool สำหรับ Function Calling ---
from typing import List, Dict, Any

# 1. กำหนด Schema ของข้อมูลที่เราต้องการ
class PhaseData(object):
    """Stores data for a single electrical phase."""
    def __init__(self, phase: str, resistance: float, impedance: float, inductance: float, phase_angle: float):
        self.phase = phase
        self.resistance = resistance
        self.impedance = impedance
        self.inductance = inductance
        self.phase_angle = phase_angle

class MotorDataTable(object):
    """A tool to structure extracted motor circuit analysis data into a table."""
    def __init__(self, data: List[PhaseData]):
        self.data = data
    
    def to_dataframe(self) -> pd.DataFrame:
        """Converts the structured data into a Pandas DataFrame."""
        records = [vars(d) for d in self.data]
        df = pd.DataFrame.from_records(records)
        # เปลี่ยนชื่อคอลัมน์ให้ตรงกับที่เราใช้
        df = df.rename(columns={
            'phase': 'Phase', # คอลัมน์แรกสำหรับแกน X
            'resistance': 'Resistance (Ohm)',
            'impedance': 'Impedance (Ohm)',
            'inductance': 'Inductance (mH)',
            'phase_angle': 'Phase Angle (°)'
        })
        return df

def extract_table_with_function_calling(text_data: str) -> pd.DataFrame:
    """
    ใช้ Gemini และ Function Calling เพื่อสกัดข้อมูลตารางเป็น DataFrame ที่แม่นยำ
    """
    # <<--- จุดแก้ไขสำคัญ ---
    # แปลง Class ของเราให้เป็น Tool ที่ Gemini เข้าใจ
    motor_data_tool = genai.protos.Tool(
        function_declarations=[
            genai.protos.FunctionDeclaration(
                name='MotorDataTable',
                description="จัดโครงสร้างข้อมูลผลการตรวจวัดมอเตอร์ให้อยู่ในรูปแบบตาราง",
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        'data': genai.protos.Schema(
                            type=genai.protos.Type.ARRAY,
                            items=genai.protos.Schema(
                                type=genai.protos.Type.OBJECT,
                                properties={
                                    'phase': genai.protos.Schema(type=genai.protos.Type.STRING),
                                    'resistance': genai.protos.Schema(type=genai.protos.Type.NUMBER),
                                    'impedance': genai.protos.Schema(type=genai.protos.Type.NUMBER),
                                    'inductance': genai.protos.Schema(type=genai.protos.Type.NUMBER),
                                    'phase_angle': genai.protos.Schema(type=genai.protos.Type.NUMBER),
                                },
                                required=['phase', 'resistance', 'impedance', 'inductance', 'phase_angle']
                            )
                        )
                    },
                    required=['data']
                )
            )
        ]
    )

    # ใช้โมเดล Pro ซึ่งเก่งเรื่อง Function Calling มากกว่า
    model = genai.GenerativeModel(
        # model_name='gemini-1.5-pro-latest', # แนะนำให้ใช้ Pro เพื่อความแม่นยำสูงสุด
        model_name='gemini-2.0-flash', # แต่ Flash ก็ใช้ได้ และเร็วกว่า
        tools=[motor_data_tool] # ส่ง Tool ที่แปลงแล้วเข้าไป
    )
    
    #---prompt เดิม---
    #จากข้อความที่ให้มาด้านล่างนี้ ให้สกัดข้อมูลผลการตรวจวัดของแต่ละเฟสออกมา
    #แล้วใช้เครื่องมือ 'MotorDataTable' เพื่อจัดโครงสร้างข้อมูลนั้น
    #กรุณาตรวจสอบข้อมูลตัวเลขให้ถูกต้องแม่นยำ อย่าปัดเศษหรือเปลี่ยนแปลงค่าเอง
        
    prompt = f"""
    คุณคือ AI ผู้ช่วยสกัดข้อมูลที่มีความเชี่ยวชาญสูง
    หน้าที่ของคุณคืออ่าน "ข้อความ" ที่ให้มาอย่างละเอียด และค้นหาตารางข้อมูลผลการตรวจวัดมอเตอร์
    **ถ้าหากคุณพบข้อมูลที่ดูเหมือนตาราง ให้คุณใช้เครื่องมือ 'MotorDataTable' เพื่อจัดโครงสร้างข้อมูลนั้นทันที**
    ค่า Resistance (Ohm), Impedance (Ohm), Inductance (mH), Phase Angle (°) มี column สุดท้ายเป็น %unbalance
    กรุณาตรวจสอบข้อมูลตัวเลขทุกตัวให้ถูกต้องแม่นยำ อย่าปัดเศษหรือเปลี่ยนแปลงค่าเอง
    ถ้าในข้อความไม่มีตารางข้อมูลที่ชัดเจน คุณไม่จำเป็นต้องเรียกใช้เครื่องมือใดๆ

    ข้อความ:
    ---
    {text_data}
    ---
    """
    
    response = model.generate_content(prompt)
    
    # ดึงผลลัพธ์จาก Tool Call
    try:
        tool_call = response.candidates[0].content.parts[0].function_call
        if tool_call.name == 'MotorDataTable':
            args = tool_call.args
            
            # แปลง list ของ dicts ที่ได้มาเป็น DataFrame
            df = pd.DataFrame(args['data'])
            
            # เปลี่ยนชื่อคอลัมน์ให้ตรงกับที่เราใช้
            df = df.rename(columns={
                'phase': 'Phase',
                'resistance': 'Resistance (Ohm)',
                'impedance': 'Impedance (Ohm)',
                'inductance': 'Inductance (mH)',
                'phase_angle': 'Phase Angle (°)'
            })
            return df
    except (IndexError, AttributeError, KeyError) as e:
        print(f"Function call failed or no tool was called: {e}")
        raise ValueError("AI ไม่สามารถสกัดข้อมูลตารางจาก PDF นี้ได้ กรุณาตรวจสอบไฟล์")

    raise ValueError("ไม่พบข้อมูลตารางที่สามารถสกัดได้")

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

def get_answer_from_manual(user_query, manual_context):
    """ฟังก์ชันสำหรับถาม-ตอบจากคู่มือ"""
    template = """
    คุณคือผู้ช่วยที่เชี่ยวชาญคู่มือการใช้งาน Motor Circuit Analyzer
    จงตอบคำถามต่อไปนี้โดยอ้างอิงจาก "บริบทจากคู่มือ" ที่ให้มาเท่านั้น
    หากคำตอบไม่ได้อยู่ในบริบท ให้ตอบว่า "ฉันไม่พบข้อมูลเกี่ยวกับเรื่องนี้ในคู่มือ"

    ---
    **[บริบทจากคู่มือการใช้งาน (Context)]**
    {manual_context}
    ---
    **[คำถาม (Question)]**
    {user_query}
    ---
    **[คำตอบ (Answer)]**
    """
    prompt = template.format(manual_context=manual_context, user_query=user_query)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    response = model.generate_content(prompt)
    return response.text

def handle_followup_question(user_question, analysis_context_str, manual_context):
    """
    ฟังก์ชันสำหรับจัดการคำถามต่อเนื่อง โดยใช้บริบทจากการวิเคราะห์ครั้งก่อน
    """
    template = """
    คุณคือ AIMCA ที่กำลังสนทนาต่อเนื่องกับผู้ใช้งานเกี่ยวกับผลการวิเคราะห์มอเตอร์ที่คุณได้ทำไปก่อนหน้านี้
    โปรดใช้ "บริบทการวิเคราะห์ครั้งก่อน" และ "บริบทจากคู่มือ" เพื่อตอบ "คำถามใหม่ของผู้ใช้" ให้กระชับและตรงประเด็น

    ---
    **[บริบทการวิเคราะห์ครั้งก่อน (Previous Analysis Context)]**
    {analysis_context}
    ---
    **[บริบทจากคู่มือ (Original Manual Context - for reference)]**
    {manual_context}
    ---
    **[คำถามใหม่ของผู้ใช้ (User's Follow-up Question)]**
    {user_question}
    ---
    **[คำตอบของคุณ (Your Answer)]**
    """
    prompt = template.format(
        analysis_context=analysis_context_str,
        manual_context=manual_context,
        user_question=user_question
    )
    
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    response = model.generate_content(prompt)
    return response.text

def create_combined_line_graph(df):
    """
    สร้างกราฟเส้นรวมที่เปรียบเทียบค่าของ 3 เฟสสำหรับทุกพารามิเตอร์
    โดยใช้แกน Y สองแกนและระบุแกน X อย่างถูกต้อง
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # <<--- จุดแก้ไขที่ 1: ระบุชื่อคอลัมน์สำหรับแกน X โดยตรง ---
    phase_column_name = 'Phase'

    # ตรวจสอบว่าคอลัมน์สำหรับแกน X มีอยู่จริงหรือไม่
    if phase_column_name not in df.columns:
        st.warning("ไม่พบคอลัมน์ 'Phase' ในข้อมูลที่สกัดได้ ไม่สามารถสร้างกราฟได้")
        return None

    params_on_left_axis = ['Resistance (Ohm)', 'Impedance (Ohm)', 'Inductance (mH)']
    param_on_right_axis = 'Phase Angle (°)'
    
    available_left_params = [p for p in params_on_left_axis if p in df.columns]
    right_axis_available = param_on_right_axis in df.columns

    if not available_left_params and not right_axis_available:
        return None

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # พล็อตเส้นกราฟสำหรับแกนซ้าย
    for param in available_left_params:
        fig.add_trace(
            go.Scatter(x=df[phase_column_name], y=df[param], name=param, mode='lines+markers'),
            secondary_y=False,
        )

    # พล็อตเส้นกราฟสำหรับแกนขวา
    if right_axis_available:
        fig.add_trace(
            go.Scatter(x=df[phase_column_name], y=df[param_on_right_axis], name=param_on_right_axis, mode='lines+markers'),
            secondary_y=True,
        )

    # ตั้งค่า Layout และชื่อแกน
    fig.update_layout(
        title_text="ภาพรวมความสัมพันธ์ระหว่างพารามิเตอร์ของแต่ละเฟส",
        legend_title="พารามิเตอร์"
    )
    fig.update_xaxes(title_text="เฟส")
    fig.update_yaxes(title_text="ค่า (Ohm, mH)", secondary_y=False)
    fig.update_yaxes(title_text="มุม (°)", secondary_y=True, showgrid=False)

    return fig

# --- 4. สร้างหน้าเว็บแอปด้วย Streamlit ---
st.set_page_config(page_title="AIMCA: ผู้ช่วยวิเคราะห์ผล MCA", layout="wide")
st.title("🤖 AIMCA: AI ผู้ช่วยวิเคราะห์ผล Motor Circuit Analyzer")

# โหลด Retriever
try:
    retriever = load_components()
except Exception as e:
    st.error(f"🚨 ไม่สามารถโหลด Vector Database หลักได้! คุณรัน ingest.py แล้วหรือยัง? ({e})")
    st.stop()

# --- UI ส่วนเลือกโหมด ---
st.sidebar.title("เมนูการทำงาน")
analysis_mode = st.sidebar.radio(
    "คุณต้องการทำอะไร?",
    (
        "วิเคราะห์ผลจากไฟล์ CSV",
        "วิเคราะห์ผลจากไฟล์ PDF",
        "สอบถามข้อมูลจากคู่มือหลัก"
    )
)

# --- โหมดที่ 1: วิเคราะห์ผลจากไฟล์ CSV ---
if analysis_mode == "วิเคราะห์ผลจากไฟล์ CSV":
    st.header("📈 วิเคราะห์ผลจากไฟล์ CSV")
    col1, col2 = st.columns([1, 2])
    with col1:
        uploaded_file = st.file_uploader("เลือกไฟล์ CSV", type="csv", key="csv_uploader", on_change=clear_analysis_state)
        user_query = st.text_area("คำสั่งเพิ่มเติม", key="csv_query")
        submit = st.button("🚀 เริ่มการวิเคราะห์", type="primary", key="csv_submit")

    # === ส่วนของ col2 ที่แก้ไขใหม่ทั้งหมด ===
    with col2:
        # 1. จัดการ Session State และรีเซ็ตเมื่อเปลี่ยนโหมด
        if 'current_mode' not in st.session_state or st.session_state.current_mode != 'csv':
            st.session_state.current_mode = 'csv'
            clear_analysis_state() # เรียกใช้ฟังก์ชันเคลียร์ค่า

        # 2. ประมวลผลไฟล์ที่อัปโหลด (ทำครั้งเดียวเมื่อไฟล์เปลี่ยน)
        if uploaded_file is not None and st.session_state.get('dataframe_for_graph') is None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.dataframe_for_graph = df # เก็บ DataFrame ไว้ใน State
            except Exception as e:
                st.error(f"ไม่สามารถอ่านไฟล์ CSV ได้: {e}")
                clear_analysis_state()

        # 3. Logic การวิเคราะห์ของ AI (เมื่อกดปุ่ม)
        if submit and uploaded_file:
            if st.session_state.dataframe_for_graph is not None:
                with st.spinner("⏳ กำลังวิเคราะห์ข้อมูล..."):
                    try:
                        df_input = st.session_state.dataframe_for_graph
                        input_data_text = df_input.to_string()
                        
                        relevant_docs = retriever.invoke(input_data_text + "\n" + user_query)
                        manual_context = "\n\n".join([doc.page_content for doc in relevant_docs])
                        
                        analysis_result = get_analysis_from_data(input_data_text, manual_context, user_query)
                        
                        st.session_state.analysis_result = analysis_result
                        st.session_state.manual_context = manual_context
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาดระหว่างการวิเคราะห์: {e}")
                        st.session_state.analysis_result = None

        # 4. แสดงผลทั้งหมด (จะแสดงก็ต่อเมื่อมีข้อมูลอยู่ใน State)
        # ส่วนแสดงกราฟเปรียบเทียบเฟส
        if st.session_state.get('dataframe_for_graph') is not None:
            st.subheader("ภาพรวมข้อมูลดิบรายเฟส")
            df_display = st.session_state.dataframe_for_graph
            fig = create_combined_line_graph(df_display) # ใช้ฟังก์ชันเดิม
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("ไม่สามารถสร้างกราฟภาพรวมได้")
            st.markdown("---")
        # ส่วนแสดงผลการวิเคราะห์ของ AI
        if st.session_state.get('analysis_result') is not None:
            analysis_result = st.session_state.analysis_result
            
            # ส่วนแสดงกราฟ
            st.subheader("📊 กราฟเปรียบเทียบผลการตรวจวัด")
            graph_data = analysis_result.get("graph_data", [])
            if graph_data:
                import plotly.graph_objects as go
                df_graph = pd.DataFrame(graph_data)
                fig = go.Figure()
                fig.add_trace(go.Bar(x=df_graph['parameter'], y=df_graph['measured_value'], name='ค่าที่วัดได้', marker_color='royalblue'))
                if 'standard_value_max' in df_graph.columns and not df_graph['standard_value_max'].isnull().all():
                    fig.add_trace(go.Bar(x=df_graph['parameter'], y=df_graph['standard_value_max'], name='เกณฑ์มาตรฐาน (สูงสุด)', marker_color='lightgrey'))
                fig.update_layout(barmode='group', title='เปรียบเทียบผลการตรวจวัดกับเกณฑ์มาตรฐาน', xaxis_title="พารามิเตอร์", yaxis_title="ค่า", legend_title="คำอธิบาย")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("ไม่สามารถสร้างกราฟได้")

            # ส่วนแสดงสรุปและคำแนะนำ
            st.subheader("📄 สรุปและคำแนะนำ")
            status = analysis_result.get('status', 'UNKNOWN').upper()
            summary = analysis_result.get('overall_summary', 'ไม่มีข้อมูลสรุป')
            if status == "GOOD":
                st.success(f"**สรุปผลโดยรวม:** {summary}")
            elif status == "WARNING":
                st.warning(f"**สรุปผลโดยรวม:** {summary}")
            elif status == "CRITICAL":
                st.error(f"**สรุปผลโดยรวม:** {summary}")
            else:
                st.info(f"**สรุปผลโดยรวม:** {summary}")

            with st.expander("ดูคำวิเคราะห์เชิงลึก"):
                st.markdown(analysis_result.get('detailed_analysis', 'ไม่มีข้อมูล'))
            with st.expander("ดูคำแนะนำ"):
                st.markdown(analysis_result.get('recommendations', 'ไม่มีข้อมูล'))

            # ส่วนของ Chatbot
            st.markdown("---")
            st.subheader("💬 คุยต่อเกี่ยวกับผลวิเคราะห์นี้")
            for role, message in st.session_state.chat_history:
                with st.chat_message(role):
                    st.markdown(message)
            if prompt := st.chat_input("ถามคำถามเพิ่มเติม..."):
                st.session_state.chat_history.append(("user", prompt))
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    with st.spinner("กำลังคิด..."):
                        analysis_context_str = json.dumps(st.session_state.analysis_result, indent=2, ensure_ascii=False)
                        response = handle_followup_question(
                            user_question=prompt,
                            analysis_context_str=analysis_context_str,
                            manual_context=st.session_state.manual_context
                        )
                        st.markdown(response)
                st.session_state.chat_history.append(("assistant", response))

        # 5. แสดงคำเตือนถ้ากดปุ่มแต่ไม่มีข้อมูลพร้อม
        elif submit:
            st.warning("กรุณาอัปโหลดไฟล์และตรวจสอบข้อมูลให้เรียบร้อยก่อนเริ่มการวิเคราะห์")


# --- โหมดที่ 2: วิเคราะห์ผลจากไฟล์ PDF ---
elif analysis_mode == "วิเคราะห์ผลจากไฟล์ PDF":
    st.header("📄 วิเคราะห์ผลจากไฟล์ PDF")
    st.info("อัปโหลดไฟล์ PDF ที่มีตารางผลการตรวจวัดเพื่อทำการวิเคราะห์")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        uploaded_file = st.file_uploader("เลือกไฟล์ PDF ผลการตรวจวัด", type="pdf", key="pdf_uploader", on_change=clear_analysis_state)
        user_query = st.text_area("คำสั่งเพิ่มเติม", key="pdf_query")
        submit = st.button("🚀 เริ่มการวิเคราะห์", type="primary", key="pdf_submit")

    # === ส่วนของ col2 ที่แก้ไขใหม่ทั้งหมด ===
    with col2:
    # 1. จัดการ Session State และรีเซ็ตเมื่อเปลี่ยนโหมด
        if 'current_mode' not in st.session_state or st.session_state.current_mode != 'pdf':
            st.session_state.current_mode = 'pdf'
            clear_analysis_state()

        # 2. ประมวลผลไฟล์ที่อัปโหลด (ทำครั้งเดียวเมื่อไฟล์เปลี่ยน)
        if uploaded_file is not None and st.session_state.get('dataframe_for_graph') is None:
            with st.spinner("AI กำลังอ่านและแปลงตารางข้อมูลจาก PDF..."):
                try:
                    # สกัดข้อความดิบ
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    loader = PyPDFLoader(tmp_file_path)
                    documents = loader.load()
                    # <<--- จุดแก้ไขที่ 1: เก็บข้อความดิบไว้ในตัวแปรใหม่ ---
                    raw_pdf_text = "\n\n".join([doc.page_content for doc in documents])
                    # <<--- จุดแก้ไขสำคัญ: เรียกใช้ฟังก์ชันใหม่ ---
                    df_from_ai = extract_table_with_function_calling(raw_pdf_text)
                                        
                    # เก็บ DataFrame และข้อความดิบลงใน State
                    st.session_state.dataframe_for_graph = df_from_ai
                    st.session_state.extracted_pdf_text = raw_pdf_text

                except Exception as e:
                    st.error(f"ไม่สามารถประมวลผลไฟล์ PDF ได้: {e}")
                    clear_analysis_state()
                finally:
                    if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                        os.remove(tmp_file_path)
        
        # 3. Logic การวิเคราะห์ของ AI (เมื่อกดปุ่ม)
        if submit and uploaded_file:
            if st.session_state.get('extracted_pdf_text') is not None:
                with st.spinner("⏳ กำลังวิเคราะห์ข้อมูล..."):
                    try:
                        # <<--- จุดแก้ไขที่ 2: ใช้ข้อความดิบในการวิเคราะห์ ---
                        input_data_for_analysis = st.session_state.extracted_pdf_text
                        
                        relevant_docs = retriever.invoke(input_data_for_analysis + "\n" + user_query)
                        manual_context = "\n\n".join([doc.page_content for doc in relevant_docs])
                        
                        analysis_result = get_analysis_from_data(input_data_for_analysis, manual_context, user_query)
                        
                        st.session_state.analysis_result = analysis_result
                        st.session_state.manual_context = manual_context
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาดระหว่างการวิเคราะห์: {e}")
                        st.session_state.analysis_result = None

        # 4. แสดงผลทั้งหมด (จะแสดงก็ต่อเมื่อมีข้อมูลอยู่ใน State)
        # ส่วนแสดงกราฟเปรียบเทียบเฟส
        if st.session_state.get('dataframe_for_graph') is not None:
            st.subheader("ภาพรวมข้อมูลดิบรายเฟส (สกัดจาก PDF)")
            df_display = st.session_state.dataframe_for_graph
            
            # สร้างกราฟ (ตอนนี้จะใช้แกน X ที่ถูกต้องแล้ว)
            fig = create_combined_line_graph(df_display) 
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                # <<--- จุดแก้ไขที่ 2: จัดเรียงคอลัมน์ของตารางก่อนแสดงผล ---
                with st.expander("ดูตารางข้อมูลที่ AI สกัดได้"):
                    # ตรวจสอบว่าคอลัมน์ 'Phase' มีอยู่จริง
                    if 'Phase' in df_display.columns:
                        # สร้าง list ของคอลัมน์ใหม่โดยเอา 'Phase' ขึ้นก่อน
                        all_columns = df_display.columns.tolist()
                        all_columns.remove('Phase')
                        new_column_order = ['Phase'] + all_columns
                        # แสดง DataFrame ที่จัดเรียงใหม่แล้ว
                        st.dataframe(df_display[new_column_order])
                    else:
                        # ถ้าไม่มี ก็แสดงตามปกติ
                        st.dataframe(df_display)
            else:
                st.write("ไม่สามารถสร้างกราฟภาพรวมได้")
            st.markdown("---")
        
        # ส่วนแสดงผลการวิเคราะห์ของ AI
        if st.session_state.get('analysis_result') is not None:
            analysis_result = st.session_state.analysis_result
            
            # ส่วนแสดงกราฟ
            st.subheader("📊 กราฟเปรียบเทียบผลการตรวจวัด")
            graph_data = analysis_result.get("graph_data", [])
            if graph_data:
                import plotly.graph_objects as go
                df_graph = pd.DataFrame(graph_data)
                fig = go.Figure()
                fig.add_trace(go.Bar(x=df_graph['parameter'], y=df_graph['measured_value'], name='ค่าที่วัดได้', marker_color='royalblue'))
                if 'standard_value_max' in df_graph.columns and not df_graph['standard_value_max'].isnull().all():
                    fig.add_trace(go.Bar(x=df_graph['parameter'], y=df_graph['standard_value_max'], name='เกณฑ์มาตรฐาน (สูงสุด)', marker_color='lightgrey'))
                fig.update_layout(barmode='group', title='เปรียบเทียบผลการตรวจวัดกับเกณฑ์มาตรฐาน', xaxis_title="พารามิเตอร์", yaxis_title="ค่า", legend_title="คำอธิบาย")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("ไม่สามารถสร้างกราฟได้")

            # ส่วนแสดงสรุปและคำแนะนำ
            st.subheader("📄 สรุปและคำแนะนำ")
            status = analysis_result.get('status', 'UNKNOWN').upper()
            summary = analysis_result.get('overall_summary', 'ไม่มีข้อมูลสรุป')
            if status == "GOOD":
                st.success(f"**สรุปผลโดยรวม:** {summary}")
            elif status == "WARNING":
                st.warning(f"**สรุปผลโดยรวม:** {summary}")
            elif status == "CRITICAL":
                st.error(f"**สรุปผลโดยรวม:** {summary}")
            else:
                st.info(f"**สรุปผลโดยรวม:** {summary}")

            with st.expander("ดูคำวิเคราะห์เชิงลึก"):
                st.markdown(analysis_result.get('detailed_analysis', 'ไม่มีข้อมูล'))
            with st.expander("ดูคำแนะนำ"):
                st.markdown(analysis_result.get('recommendations', 'ไม่มีข้อมูล'))
            
            if st.session_state.get('extracted_pdf_text'):
                with st.expander("ข้อความที่สกัดได้จาก PDF ที่อัปโหลด"):
                    st.text(st.session_state.extracted_pdf_text)

            # ส่วนของ Chatbot
            st.markdown("---")
            st.subheader("💬 คุยต่อเกี่ยวกับผลวิเคราะห์นี้")
            for role, message in st.session_state.chat_history:
                with st.chat_message(role):
                    st.markdown(message)
            if prompt := st.chat_input("ถามคำถามเพิ่มเติม..."):
                st.session_state.chat_history.append(("user", prompt))
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    with st.spinner("กำลังคิด..."):
                        analysis_context_str = json.dumps(st.session_state.analysis_result, indent=2, ensure_ascii=False)
                        response = handle_followup_question(
                            user_question=prompt,
                            analysis_context_str=analysis_context_str,
                            manual_context=st.session_state.manual_context
                        )
                        st.markdown(response)
                st.session_state.chat_history.append(("assistant", response))

        # 4. แสดงคำเตือนถ้ากดปุ่มแต่ไม่มีไฟล์
        elif submit:
            st.warning("กรุณาอัปโหลดไฟล์ PDF ก่อนครับ")


# --- โหมดที่ 3: สอบถามข้อมูลจากคู่มือหลัก ---
elif analysis_mode == "สอบถามข้อมูลจากคู่มือหลัก":
    st.header("❓ สอบถามข้อมูลจากคู่มือหลัก")
    user_query = st.text_input("คำถามของคุณ:", key="manual_query", placeholder="เช่น: ค่าความต้านทานที่ยอมรับได้ควรเป็นเท่าไหร่?")
    if user_query:
        with st.spinner("⏳ กำลังค้นหาคำตอบ..."):
            relevant_docs = retriever.invoke(user_query)
            manual_context = "\n\n".join([doc.page_content for doc in relevant_docs])
            answer = get_answer_from_manual(user_query, manual_context)
            st.subheader("คำตอบ:")
            st.markdown(answer)