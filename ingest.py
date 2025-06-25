# ingest.py - โปรแกรมสำหรับประมวลผล PDF และสร้างฐานข้อมูลความรู้

# 1. Import ไลบรารีที่จำเป็น
import time
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 2. กำหนดค่าต่างๆ
PDF_DATA_PATH = "data/"
VECTOR_DB_PATH = "vectorstore/db_chroma"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # โมเดล Embedding ขนาดเล็กและเร็ว

def create_vector_db():
    """
    ฟังก์ชันหลักในการสร้าง Vector Database จากไฟล์ PDF
    """
    print("--------------------------------------------------")
    print("🚀 เริ่มกระบวนการสร้างฐานข้อมูลความรู้ (Vector DB)...")
    
    # โหลดเอกสาร PDF ทั้งหมดจากโฟลเดอร์ data
    print(f"📚 กำลังโหลดไฟล์ PDF จากโฟลเดอร์ '{PDF_DATA_PATH}'...")
    start_time = time.time()
    loader = DirectoryLoader(PDF_DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    if not documents:
        print("🚨 ไม่พบไฟล์ PDF ในโฟลเดอร์ 'data'. กรุณาตรวจสอบและลองอีกครั้ง")
        return
    end_time = time.time()
    print(f"✅ โหลดเอกสารสำเร็จ! พบ {len(documents)} ไฟล์. (ใช้เวลา: {end_time - start_time:.2f} วินาที)")

    # แบ่งเอกสารเป็นส่วนๆ (Chunking)
    print("쪼개기 กำลังแบ่งเอกสารเป็นส่วนย่อย (Chunks)...")
    start_time = time.time()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    end_time = time.time()
    print(f"✅ แบ่งเอกสารเป็น {len(texts)} chunks สำเร็จ. (ใช้เวลา: {end_time - start_time:.2f} วินาที)")

    # สร้าง Embedding Model (จะดาวน์โหลดอัตโนมัติในครั้งแรก)
    print(f"🧠 กำลังโหลด Embedding Model ({EMBEDDING_MODEL_NAME})...")
    start_time = time.time()
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'} # บังคับให้ใช้ CPU
    )
    end_time = time.time()
    print(f"✅ โหลดโมเดลสำเร็จ. (ใช้เวลา: {end_time - start_time:.2f} วินาที)")

    # สร้างและจัดเก็บ Vector ลงใน ChromaDB
    print(f"💾 กำลังสร้างและจัดเก็บข้อมูลลงใน Vector DB ที่ '{VECTOR_DB_PATH}'...")
    start_time = time.time()
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embedding_model,
        persist_directory=VECTOR_DB_PATH
    )
    end_time = time.time()
    print(f"✅ สร้างฐานข้อมูล Vector DB สำเร็จ! (ใช้เวลา: {end_time - start_time:.2f} วินาที)")
    print("--------------------------------------------------")
    print("🎉 กระบวนการทั้งหมดเสร็จสมบูรณ์! ฐานข้อมูลของคุณพร้อมใช้งานแล้ว")
    print("--------------------------------------------------")

if __name__ == "__main__":
    create_vector_db()