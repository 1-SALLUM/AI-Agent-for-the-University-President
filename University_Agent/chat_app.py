import streamlit as st
import ollama
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="مساعد مدير الجامعة", page_icon="🎓")
st.title("🎓 المساعد التنفيذي الذكي لمكتب مدير الجامعة")
st.write("مرحباً بك. أنا مساعدك الذكي المربوط بأنظمة ولوائح الجامعة.")

@st.cache_resource
def load_university_data():
    docx_loader = DirectoryLoader("university_docs", glob="**/*.docx", loader_cls=Docx2txtLoader)
    docx_docs = docx_loader.load()
    
    pdf_loader = DirectoryLoader("university_docs", glob="**/*.pdf", loader_cls=PyPDFLoader)
    pdf_docs = pdf_loader.load()
    
    all_docs = docx_docs + pdf_docs
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(all_docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

vectorstore = load_university_data()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_question = st.chat_input("اطرح سؤالك هنا...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        with st.spinner("جاري التحليل والبحث في اللوائح..."):
            
            history_context = ""
            if len(st.session_state.messages) > 1:
                recent_msgs = st.session_state.messages[-4:-1] 
                history_context = "\n".join([f"{m['role']}: {m['content']}" for m in recent_msgs])
            
            query_generator_prompt = f"""أنت خبير في صياغة استعلامات البحث (Search Queries) لأنظمة RAG الإدارية في جامعة الإمام.
            مهمتك: تحويل سؤال المستخدم العامي، أو المركب، أو المعتمد على السياق إلى 3 استعلامات بحث باللغة العربية الفصحى الرسمية لتناسب البحث في اللوائح.

            أمثلة للتعلم:
            - سؤال المستخدم: "هل السكيورتي عندهم أوف بكرة؟"
              الاستعلامات:
              ما هو وضع موظفي الأمن والسلامة من قرار تعليق الدراسة؟
              هل يشمل استثناء الحضور موظفي الحراسات الأمنية؟
              دوام حراس الأمن وموظفي الصيانة أثناء سوء الأحوال الجوية.

            - سؤال المستخدم: "هل القرار يشمل الإداريين، وكيف وضع العملي؟"
              الاستعلامات:
              هل يشمل قرار تعليق الدراسة الموظفين الإداريين؟
              ما هو الإجراء المتبع بخصوص المحاضرات العملية والمختبرات؟
              القرارات الخاصة بداوم الإداريين والمحاضرات العملية.

            الآن، قم بتطبيق هذا المنطق الاحترافي على السؤال التالي:
            سياق المحادثة السابقة:
            {history_context if history_context else 'لا يوجد محادثة سابقة'}

            سؤال المستخدم الحالي: {user_question}

            أخرج 3 استعلامات بحث فقط، كل استعلام في سطر منفصل، بدون أي ترقيم، أو رموز، أو مقدمات نصية نهائياً:"""
            
            mq_response = ollama.chat(model='llama3.1', messages=[{'role': 'user', 'content': query_generator_prompt}], options={'temperature': 0.1})
            generated_queries = mq_response['message']['content'].strip().split('\n')
            
            all_queries = [user_question] + [q.strip() for q in generated_queries if q.strip()]
            
            unique_docs = []
            seen_content = set()
            for query in all_queries:
                # التعديل الأول: زيادة k إلى 3 لضمان اصطياد كل الفقرات للأسئلة المزدوجة والمكانية
                docs = vectorstore.similarity_search(query, k=5)
                for d in docs:
                    if d.page_content not in seen_content:
                        seen_content.add(d.page_content)
                        unique_docs.append(d.page_content)
            
            # التعديل الثاني: السماح بأفضل 5 فقرات لتكوين سياق أوسع للنموذج
            best_paragraph = "\n\n---\n\n".join(unique_docs[:5])
            
            # تعليمات مبسطة وصارمة جداً تناسب قدرات النماذج المحلية
            system_instruction = """
            أنت مساعد إداري لجامعة الإمام. أجب مباشرة وبدون أي مقدمات (لا تقل "حسناً" أو "الإجابة هي").

            القواعد الصارمة:
            1. اعتمد فقط على النصوص المرفقة أدناه.
            2. إذا سأل المستخدم عن معلومة، والنص يحتوي على معلومة مشابهة ولكن مختلفة (مثل لغة أخرى)، اذكر الموجود في النص فقط لتصحيح المفهوم.
            3. إذا كان جزء من السؤال غير موجود في النص المرفق (مثل رقم بوابة، أو رسوم، أو فئة معينة)، يُمنع منعاً باتاً استنتاجه أو اختراعه. يجب أن تقول صراحة: "لا توجد معلومات في التعميم عن [اذكر الشيء المفقود]".
            4. لا تحاول إرضاء المستخدم باختراع إجابات. الاعتذار عن غياب المعلومة هو التصرف الصحيح.
            """
            
            prompt_content = f"النصوص المرجعية:\n{best_paragraph}\n\nالسؤال: {user_question}"
            
            final_response = ollama.chat(model='llama3.1', messages=[
                {'role': 'system', 'content': system_instruction},
                {'role': 'user', 'content': prompt_content}
            ], options={'temperature': 0.2}) # خفضنا الحرارة قليلاً لتفادي الهلوسة في الاستنتاج
            
            assistant_reply = final_response['message']['content']
            
            st.write(assistant_reply)
            
            with st.expander("🔍 كواليس معالجة الاستعلام (RAG Debugger)"):
                st.write("**السياق الذي فهمه النظام من الذاكرة:**")
                st.caption(history_context if history_context else "لا يوجد")
                st.write("**استعلامات البحث التي تم ترجمتها وتوليدها:**")
                for q in all_queries:
                    st.caption(f"- {q}")
                st.write("**النصوص التي تم العثور عليها:**")
                st.info(best_paragraph)
            
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})