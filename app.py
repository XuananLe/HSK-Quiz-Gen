import streamlit as st
from openai import OpenAI
import instructor
import os
import base64
from PIL import Image
import io
import re
import random
from typing import List, Optional, Union
from pydantic import BaseModel, Field, field_validator
from enum import Enum

def image_to_base64(image_file) -> str:
    """Convert uploaded image to base64"""
    return base64.b64encode(image_file.read()).decode("utf-8")

# Initialize OpenAI client with instructor using Streamlit secrets
try:
    api_key = st.secrets["openai"]["api_key"]
except KeyError:
    st.error("❌ OpenAI API key not found! Please configure it in Streamlit secrets.")
    st.stop()

client = instructor.from_openai(OpenAI(api_key=api_key))

# Pydantic Models for structured data
class QuestionType(str, Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    TEXT_INPUT = "text_input"
    TRUE_FALSE = "true_false"

class QuizQuestion(BaseModel):
    """Structured representation of a quiz question"""
    id: int = Field(..., description="Question ID")
    type: str = Field(..., description="Question type/category")
    question: str = Field(..., description="The question text")
    chinese_word: str = Field(..., description="The Chinese word/character")
    pinyin: str = Field(..., description="Pinyin pronunciation")
    meaning: str = Field(..., description="Vietnamese/English meaning")
    wrong_meanings: List[str] = Field(default=[], description="Wrong meaning options for multiple choice")
    explanation: Optional[str] = Field(default="", description="Additional explanation")
    
    @field_validator('question')
    def question_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Question text cannot be empty')
        return v.strip()
    
    @field_validator('chinese_word')
    def chinese_word_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Chinese word cannot be empty')
        return v.strip()
    
    @field_validator('pinyin')
    def pinyin_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Pinyin cannot be empty')
        return v.strip()
    
    @field_validator('meaning')
    def meaning_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Meaning cannot be empty')
        return v.strip()
    
    @field_validator('wrong_meanings')
    def validate_wrong_meanings(cls, v):
        # Clean up wrong meanings
        cleaned_meanings = [meaning.strip() for meaning in v if meaning.strip()]
        return cleaned_meanings
    
    @property
    def all_meaning_options(self) -> List[str]:
        """Get all meaning options (correct + wrong) shuffled"""
        options = [self.meaning] + self.wrong_meanings
        random.shuffle(options)
        return options

class QuizData(BaseModel):
    questions: List[QuizQuestion] = Field(default=[], description="List of quiz questions")
    title: Optional[str] = Field(default="Quiz", description="Quiz title")

    @field_validator('questions')
    def must_have_questions(cls, v):
        if not v:
            raise ValueError('Quiz must have at least one question')
        return v
    
    def __len__(self):
        return len(self.questions)
    
    def get_question(self, index: int) -> Optional[QuizQuestion]:
        """Get question by index"""
        if 0 <= index < len(self.questions):
            return self.questions[index]
        return None

def generate_quiz_from_images(image_files):
    """Generate quiz from multiple images"""
    all_quiz_data = []
    
    for i, image_file in enumerate(image_files):
        try:
            base64_img = image_to_base64(image_file)
            
            quiz_data = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.7,
                max_tokens=2048,
                response_model=QuizData,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Bạn là giáo viên tiếng Trung. Đây là ảnh chụp một trang ghi chú gồm từ Hán, pinyin, loại từ, và nghĩa tiếng Việt. "
                                    "Hãy tạo 8–12 câu hỏi quiz theo format sau: "
                                    "- Hiển thị từ tiếng Trung (Hán tự) "
                                    "- Học sinh cần điền pinyin (text input) "
                                    "- Học sinh chọn nghĩa đúng từ các lựa chọn (multiple choice) "
                                    "Trả về dữ liệu có cấu trúc với: "
                                    "- title: tên bài quiz (dựa trên nội dung ảnh) "
                                    "- questions: danh sách câu hỏi, mỗi câu có: "
                                    "  * id: số thứ tự "
                                    "  * type: 'chinese_to_pinyin_meaning' "
                                    "  * question: câu hỏi dạng 'Pinyin và nghĩa của từ [từ tiếng Trung] là gì?' "
                                    "  * chinese_word: từ tiếng Trung (hiển thị cho học sinh) "
                                    "  * pinyin: cách đọc pinyin đúng "
                                    "  * meaning: nghĩa tiếng Việt đúng "
                                    "  * wrong_meanings: 3-4 nghĩa sai để tạo multiple choice "
                                    "  * explanation: giải thích thêm về từ (nếu có) "
                                    "Ví dụ: question='Pinyin và nghĩa của từ 学生 là gì?', chinese_word='学生', pinyin='xuéshēng', meaning='học sinh', wrong_meanings=['giáo viên', 'bạn bè', 'gia đình']"
                                )
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_img}"
                                }
                            }
                        ]
                    }
                ]
            )
            all_quiz_data.append(quiz_data)
        except Exception as e:
            # Create error question for this image
            error_question = QuizQuestion(
                id=1,
                type="error",
                question=f"Lỗi khi tạo quiz từ ảnh {i+1}: {str(e)}",
                chinese_word="N/A",
                pinyin="N/A",
                meaning="N/A",
                wrong_meanings=[]
            )
            error_quiz = QuizData(questions=[error_question], title=f"Lỗi - Ảnh {i+1}")
            all_quiz_data.append(error_quiz)
    
    # Combine all quiz data into one
    if not all_quiz_data:
        return None
    
    combined_questions = []
    question_id = 1
    titles = []
    
    for quiz_data in all_quiz_data:
        titles.append(quiz_data.title)
        for question in quiz_data.questions:
            question.id = question_id
            combined_questions.append(question)
            question_id += 1
    
    combined_title = f"Quiz tổng hợp từ {len(image_files)} ảnh: " + ", ".join(titles[:3])
    if len(titles) > 3:
        combined_title += f" và {len(titles) - 3} ảnh khác"
    
    return QuizData(questions=combined_questions, title=combined_title)

def generate_quiz_from_image(image_file):
    """Generate quiz from single image (kept for backward compatibility)"""
    return generate_quiz_from_images([image_file])

def display_question(question: QuizQuestion, question_num: int) -> bool:
    """Display a question showing Chinese word and asking for pinyin + meaning"""
    st.subheader(f"Câu hỏi {question_num}")
    st.write(f"**Loại:** {question.type}")
    
    # Display the Chinese word prominently
    st.markdown(f"### 🇨🇳 **{question.chinese_word}**")
    st.write(f"**Câu hỏi:** {question.question}")
    
    # Create two columns for pinyin input and meaning selection
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**1. Nhập pinyin:**")
        user_pinyin = st.text_input(
            "Pinyin:",
            key=f"pinyin_{question.id}",
            placeholder="Ví dụ: xuéshēng",
            label_visibility="collapsed"
        )
    
    with col2:
        st.write("**2. Chọn nghĩa đúng:**")
        # Get shuffled options for this question instance
        if f"options_{question.id}" not in st.session_state:
            st.session_state[f"options_{question.id}"] = question.all_meaning_options
        
        meaning_options = st.session_state[f"options_{question.id}"]
        user_meaning = st.radio(
            "Nghĩa:",
            meaning_options,
            key=f"meaning_{question.id}",
            label_visibility="collapsed"
        )
    
    correct = False
    
    if st.button(f"Kiểm tra câu {question_num}", key=f"check_{question.id}"):
        # Check both pinyin and meaning
        pinyin_correct = user_pinyin.strip().lower() == question.pinyin.strip().lower()
        meaning_correct = user_meaning == question.meaning
        
        if pinyin_correct and meaning_correct:
            st.success("✅ Đúng hoàn toàn! Cả pinyin và nghĩa đều chính xác!")
            correct = True
        elif pinyin_correct and not meaning_correct:
            st.warning(f"⚠️ Pinyin đúng rồi! Nhưng nghĩa sai. Nghĩa đúng là: **{question.meaning}**")
        elif not pinyin_correct and meaning_correct:
            st.warning(f"⚠️ Nghĩa đúng rồi! Nhưng pinyin sai. Pinyin đúng là: **{question.pinyin}**")
        else:
            st.error("❌ Cả hai đều sai!")
        
        # Always show the complete answer after checking
        st.info("📖 **Đáp án đầy đủ:**")
        ans_col1, ans_col2, ans_col3 = st.columns(3)
        with ans_col1:
            st.write(f"**Hán tự:** {question.chinese_word}")
        with ans_col2:
            if pinyin_correct:
                st.write(f"**Pinyin:** ✅ {question.pinyin}")
            else:
                st.write(f"**Pinyin:** ❌ {question.pinyin}")
        with ans_col3:
            if meaning_correct:
                st.write(f"**Nghĩa:** ✅ {question.meaning}")
            else:
                st.write(f"**Nghĩa:** ❌ {question.meaning}")
        
        if question.explanation:
            st.info(f"💡 **Giải thích:** {question.explanation}")
    
    st.write("---")
    return correct

def main():
    st.set_page_config(
        page_title="Quiz Generator - Tiếng Trung",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Quiz Generator - Tiếng Trung")
    st.write("Tải lên một hoặc nhiều hình ảnh ghi chú tiếng Trung để tạo quiz tương tác!")
    st.write("**Format:** Hiển thị từ Hán → Học sinh điền pinyin + chọn nghĩa")
    st.info("💡 **Mẹo:** Bạn có thể tải lên nhiều ảnh cùng lúc để tạo một bài quiz lớn từ nhiều trang ghi chú!")
    
    # Sidebar for file upload
    st.sidebar.header("🖼️ Tải lên hình ảnh")
    uploaded_files = st.sidebar.file_uploader(
        "Chọn file hình ảnh (có thể chọn nhiều file)",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Tải lên một hoặc nhiều hình ảnh chứa từ vựng tiếng Trung, pinyin và nghĩa tiếng Việt"
    )
    
    # Option to use default image
    use_default = st.sidebar.checkbox("Sử dụng hình ảnh mẫu (A.jpg)")
    
    if uploaded_files or use_default:
        # Display uploaded images
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🖼️ Hình ảnh đã tải")
            image_files = []
            
            if uploaded_files:
                for i, uploaded_file in enumerate(uploaded_files):
                    image = Image.open(uploaded_file)
                    st.image(image, caption=f"Ảnh {i+1}: {uploaded_file.name}", use_column_width=True)
                    image_files.append(uploaded_file)
                st.write(f"**Tổng cộng: {len(uploaded_files)} ảnh**")
            elif use_default:
                try:
                    with open("A.jpg", "rb") as f:
                        image = Image.open("A.jpg")
                        st.image(image, caption="Hình ảnh mẫu (A.jpg)", use_column_width=True)
                        # Reset file pointer for processing
                        image_files = [open("A.jpg", "rb")]
                except FileNotFoundError:
                    st.error("Không tìm thấy file A.jpg!")
                    return
        
        with col2:
            # Generate quiz button
            if st.button("🎯 Tạo Quiz", type="primary"):
                with st.spinner("Đang tạo quiz... Vui lòng đợi!"):
                    try:
                        # Reset file pointers if using uploaded files
                        if uploaded_files:
                            for uploaded_file in uploaded_files:
                                uploaded_file.seek(0)
                            quiz_data = generate_quiz_from_images(uploaded_files)
                        else:
                            quiz_data = generate_quiz_from_images(image_files)
                        
                        if quiz_data is None:
                            st.error("❌ Không thể tạo quiz từ các ảnh đã tải!")
                            return
                        
                        # Store quiz data in session state
                        st.session_state.quiz_data = quiz_data
                        st.session_state.current_question = 0
                        st.session_state.score = 0
                        st.session_state.answered_questions = set()
                        
                        # Clear any previous question options
                        for key in list(st.session_state.keys()):
                            if key.startswith("options_"):
                                del st.session_state[key]
                        
                        num_images = len(uploaded_files) if uploaded_files else 1
                        st.success(f"✅ Quiz đã được tạo từ {num_images} ảnh với tổng cộng {len(quiz_data.questions)} câu hỏi!")
                        
                    except Exception as e:
                        st.error(f"❌ Lỗi khi tạo quiz: {str(e)}")
    
    # Display quiz if available
    if 'quiz_data' in st.session_state:
        quiz_data = st.session_state.quiz_data
        
        st.header(f"📝 {quiz_data.title}")
        st.write(f"**Tổng số câu hỏi:** {len(quiz_data.questions)}")
        
        # Quiz navigation
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("⬅️ Câu trước") and st.session_state.current_question > 0:
                st.session_state.current_question -= 1
                st.rerun()
        
        with col2:
            question_options = [f"Câu {i+1}" for i in range(len(quiz_data.questions))]
            selected_q = st.selectbox(
                "Chọn câu hỏi:",
                question_options,
                index=st.session_state.current_question
            )
            st.session_state.current_question = question_options.index(selected_q)
        
        with col3:
            if st.button("Câu sau ➡️") and st.session_state.current_question < len(quiz_data.questions) - 1:
                st.session_state.current_question += 1
                st.rerun()
        
        # Display current question
        current_q_idx = st.session_state.current_question
        current_question = quiz_data.questions[current_q_idx]
        
        # Display question
        with st.container():
            is_correct = display_question(current_question, current_q_idx + 1)
            
            # Update score tracking
            if is_correct and current_q_idx not in st.session_state.answered_questions:
                st.session_state.score += 1
                st.session_state.answered_questions.add(current_q_idx)
        
        # Progress and score display
        st.sidebar.header("📊 Tiến độ")
        progress = len(st.session_state.answered_questions) / len(quiz_data.questions)
        st.sidebar.progress(progress)
        st.sidebar.write(f"Đã trả lời: {len(st.session_state.answered_questions)}/{len(quiz_data.questions)}")
        st.sidebar.write(f"Điểm số: {st.session_state.score}/{len(st.session_state.answered_questions) if st.session_state.answered_questions else 0}")
        
        # Download quiz option
        st.sidebar.header("💾 Tải xuống")
        if st.sidebar.button("📄 Tải quiz dạng text"):
            quiz_text = f"QUIZ TITLE: {quiz_data.title}\n"
            quiz_text += f"Total Questions: {len(quiz_data.questions)}\n\n"
            
            for i, question in enumerate(quiz_data.questions, 1):
                quiz_text += f"Question {i}:\n"
                quiz_text += f"Type: {question.type}\n"
                quiz_text += f"Chinese Word: {question.chinese_word}\n"
                quiz_text += f"Question: {question.question}\n"
                quiz_text += f"Correct Pinyin: {question.pinyin}\n"
                quiz_text += f"Correct Meaning: {question.meaning}\n"
                quiz_text += f"Wrong Options: {', '.join(question.wrong_meanings)}\n"
                
                if question.explanation:
                    quiz_text += f"Explanation: {question.explanation}\n"
                
                quiz_text += "-" * 40 + "\n"
            
            st.sidebar.download_button(
                label="📥 Download Quiz",
                data=quiz_text,
                file_name="quiz_output.txt",
                mime="text/plain"
            )
    else:
        st.info("👆 Tải lên hình ảnh và nhấn 'Tạo Quiz' để bắt đầu!")

if __name__ == "__main__":
    main()
