import ollama
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import warnings
warnings.filterwarnings("ignore") # هذا السطر لإخفاء التنبيهات المزعجة التي ظهرت لك سابقاً

# 1. قراءة ملف الجامعة
loader = Docx2txtLoader("letter.docx")
docs = loader.load()

# 2. تقطيع النص إلى فقرات صغيرة (مثلاً كل 500 حرف تعتبر فقرة)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

print("جاري تحويل الملفات إلى قاعدة بيانات ذكية (قد يستغرق التحميل دقيقة في المرة الأولى فقط)...")

# 3. تحميل نموذج مخصص للغة العربية لتحويل الكلمات إلى أرقام، وتخزينها في قاعدة بيانات Chroma
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# 4. سؤال مدير الجامعة (نبحث عن معلومة محددة جداً)
user_question = "هل حراس الأمن يشملهم قرار تعليق الدراسة؟"

print("جاري البحث في الملفات عن الإجابة...")

# 5. النظام يبحث في قاعدة البيانات عن الفقرة التي تحتوي على الإجابة فقط
relevant_docs = vectorstore.similarity_search(user_question)
best_paragraph = relevant_docs[0].page_content # أخذنا أقرب فقرة للسؤال

# 6. نرسل الفقرة والسؤال للمساعد الذكي Qwen
final_prompt = f"بناءً على هذه الفقرة من قرارات الجامعة: '{best_paragraph}'، أجب عن هذا السؤال بدقة واختصار: {user_question}"

response = ollama.chat(model='qwen2.5:7b', messages=[
  {
    'role': 'user',
    'content': final_prompt,
  },
])

print("\nإجابة المساعد الذكي:")
print(response['message']['content'])