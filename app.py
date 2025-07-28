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
import concurrent.futures
import time

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
    GAP_FILLING = "gap_filling"
    DIALOGUE_ARRANGEMENT = "dialogue_arrangement"
    READING_COMPREHENSION = "reading_comprehension"
    CHINESE_TO_PINYIN_MEANING = "chinese_to_pinyin_meaning"

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
    
    # Fields for gap filling questions
    context_sentence: Optional[str] = Field(default="", description="Full sentence with gap for gap filling")
    options: List[str] = Field(default=[], description="Options for gap filling or dialogue arrangement")
    correct_answer: Optional[str] = Field(default="", description="Correct answer for gap filling")
    hsk_level: Optional[int] = Field(default=4, description="HSK level of the vocabulary used")
    
    # Fields for dialogue arrangement
    dialogue_parts: List[str] = Field(default=[], description="Parts of dialogue for arrangement")
    correct_order: List[int] = Field(default=[], description="Correct order of dialogue parts")
    
    # Fields for reading comprehension
    reading_text: Optional[str] = Field(default="", description="Text for reading comprehension")
    subquestions: List[str] = Field(default=[], description="Questions about the reading text")
    subanswers: List[str] = Field(default=[], description="Answers to the subquestions")
    suboptions: List[List[str]] = Field(default=[], description="Options for each subquestion")
    
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
        # Clean up wrong meanings and remove duplicates
        cleaned_meanings = []
        seen = set()
        for meaning in v:
            meaning_clean = meaning.strip()
            if meaning_clean and meaning_clean.lower() not in seen:
                cleaned_meanings.append(meaning_clean)
                seen.add(meaning_clean.lower())
        return cleaned_meanings
    
    @property
    def all_meaning_options(self) -> List[str]:
        """Get all meaning options (correct + wrong) shuffled"""
        # Ensure we always have at least 3 wrong meanings
        if len(self.wrong_meanings) < 3:
            backup_options = ["học sinh", "giáo viên", "bạn bè", "gia đình", "thời gian", "từ khác"]
            additional_options = []
            for option in backup_options:
                if len(self.wrong_meanings) + len(additional_options) >= 3:
                    break
                if option.lower() not in [m.lower() for m in self.wrong_meanings] and option.lower() != self.meaning.lower():
                    additional_options.append(option)
            
            # Create and shuffle options
            options = [self.meaning] + self.wrong_meanings + additional_options
        else:
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


def generate_single_quiz(image_file, image_index):
    """Generate quiz from a single image - helper function for parallel processing"""
    try:
        # Reset file pointer if needed
        if hasattr(image_file, 'seek'):
            image_file.seek(0)
            
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
                                "Hãy tạo 8-12 câu hỏi quiz với các dạng bài tập khác nhau: "
                                "- Dạng 1: Hiển thị từ tiếng Trung → Học sinh điền pinyin + chọn nghĩa đúng (4-6 câu)"
                                "- Dạng 2: Điền từ vào chỗ trống, sử dụng từ vựng ở HSK level 4 (1-2 câu)"
                                "- Dạng 3: Sắp xếp hội thoại theo thứ tự đúng (0-1 câu)"
                                "- Dạng 4: Đọc hiểu văn bản ngắn và trả lời câu hỏi (0-1 câu)"
                                "Trả về dữ liệu có cấu trúc với: "
                                "- title: tên bài quiz (dựa trên nội dung ảnh) "
                                "- questions: danh sách câu hỏi, mỗi câu có các trường tùy theo loại câu hỏi: "
                                ""
                                "1. Đối với câu hỏi pinyin và nghĩa (type: 'chinese_to_pinyin_meaning'):"
                                "  * id: số thứ tự "
                                "  * type: 'chinese_to_pinyin_meaning' "
                                "  * question: câu hỏi dạng 'Pinyin và nghĩa của từ [từ tiếng Trung] là gì?' "
                                "  * chinese_word: từ tiếng Trung (hiển thị cho học sinh) "
                                "  * pinyin: cách đọc pinyin đúng "
                                "  * meaning: nghĩa tiếng Việt đúng "
                                "  * wrong_meanings: 3-4 nghĩa sai để tạo multiple choice "
                                ""
                                "2. Đối với câu hỏi điền vào chỗ trống (type: 'gap_filling'):"
                                "  * id: số thứ tự"
                                "  * type: 'gap_filling'"
                                "  * question: Câu hỏi dạng 'Chọn từ phù hợp để điền vào chỗ trống'"
                                "  * context_sentence: Câu hoàn chỉnh với '___ ' là chỗ cần điền"
                                "  * options: 4 lựa chọn từ để điền"
                                "  * correct_answer: Từ đúng để điền vào chỗ trống"
                                "  * chinese_word: Từ đúng để điền (giống correct_answer)"
                                "  * pinyin: Pinyin của từ đúng"
                                "  * meaning: Nghĩa của từ đúng"
                                "  * hsk_level: 4 (hoặc cấp độ HSK của từ vựng)"
                                ""
                                "3. Đối với câu hỏi sắp xếp hội thoại (type: 'dialogue_arrangement'):"
                                "  * id: số thứ tự"
                                "  * type: 'dialogue_arrangement'"
                                "  * question: 'Sắp xếp các phần của hội thoại theo thứ tự đúng'"
                                "  * dialogue_parts: Mảng các phần của hội thoại (3-5 phần)"
                                "  * correct_order: Mảng các số nguyên thể hiện thứ tự đúng [0, 1, 2, ...]"
                                "  * chinese_word: Chủ đề của hội thoại"
                                "  * pinyin: Pinyin của chủ đề"
                                "  * meaning: Nghĩa của chủ đề"
                                ""
                                "4. Đối với câu hỏi đọc hiểu (type: 'reading_comprehension'):"
                                "  * id: số thứ tự"
                                "  * type: 'reading_comprehension'"
                                "  * question: '阅读理解 (Đọc hiểu)'"
                                "  * reading_text: Nội dung đoạn văn dài (150-250 từ) bằng tiếng Trung (HSK 4-5), có thể có đoạn văn đối thoại hoặc văn xuôi"
                                "  * subquestions: Mảng các câu hỏi con bằng tiếng Trung về nội dung đoạn văn (2-4 câu hỏi)"
                                "  * suboptions: Mảng các mảng lựa chọn bằng tiếng Trung cho từng câu hỏi con"
                                "  * subanswers: Mảng các đáp án đúng cho từng câu hỏi con"
                                "  * chinese_word: Tiêu đề của đoạn văn"
                                "  * pinyin: Pinyin của tiêu đề"
                                "  * meaning: Nghĩa của tiêu đề"
                                "  * explanation: Giải thích các từ khó hoặc ngữ pháp phức tạp trong bài đọc"
                                "  * explanation: giải thích thêm về từ (nếu có) "
                                "Lưu ý quan trọng:"
                                "1. Mỗi câu hỏi phải có wrong_meanings riêng biệt, không được sử dụng lại wrong_meanings ở các câu khác"
                                "2. Các nghĩa sai phải khác hoàn toàn với nghĩa đúng"
                                "3. Hãy tạo nghĩa sai liên quan đến ngữ cảnh của từ, không chỉ chọn nghĩa ngẫu nhiên"
                                "4. Tránh sử dụng lại các nghĩa sai giống nhau ở các câu hỏi khác nhau"
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
        return quiz_data, image_index, None
    except Exception as e:
        # Create error question for this image
        error_question = QuizQuestion(
            id=1,
            type="error",
            question=f"Lỗi khi tạo quiz từ ảnh {image_index+1}: {str(e)}",
            chinese_word="N/A",
            pinyin="N/A",
            meaning="N/A",
            wrong_meanings=[]
        )
        error_quiz = QuizData(questions=[error_question], title=f"Lỗi - Ảnh {image_index+1}")
        return error_quiz, image_index, str(e)


def generate_quiz_from_images(image_files):
    """Generate quiz from multiple images using parallel processing"""
    if not image_files:
        return None
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(6, len(image_files))) as executor:
        future_to_index = {
            executor.submit(generate_single_quiz, image_file, i): i 
            for i, image_file in enumerate(image_files)
        }
        
        results = {}
        for future in concurrent.futures.as_completed(future_to_index):
            quiz_data, image_index, error = future.result()
            results[image_index] = quiz_data
    
    # Sort results by original image order
    all_quiz_data = []
    for i in range(len(image_files)):
        if i in results:
            all_quiz_data.append(results[i])
    
    # Combine all quiz data into one
    if not all_quiz_data:
        return None
    
    combined_questions = []
    question_id = 1
    titles = []
    
    # Keep track of used wrong meanings to avoid duplicates
    used_wrong_meanings = set()
    
    for quiz_data in all_quiz_data:
        titles.append(quiz_data.title)
        for question in quiz_data.questions:
            # Ensure wrong_meanings don't overlap with previously used ones
            filtered_wrong_meanings = []
            for meaning in question.wrong_meanings:
                if meaning.lower() not in used_wrong_meanings and meaning.lower() != question.meaning.lower():
                    filtered_wrong_meanings.append(meaning)
                    used_wrong_meanings.add(meaning.lower())
            
            # Make sure we have at least 3 wrong meanings
            if len(filtered_wrong_meanings) < 3:
                # Generate some new meanings that are different from the correct one
                # and from each other
                common_wrong_options = [
                    "học sinh", "giáo viên", "bạn bè", "gia đình", "cuộc sống",
                    "công việc", "thời gian", "nhà cửa", "tình yêu", "thức ăn",
                    "nước uống", "sức khỏe", "tiền bạc", "giao thông", "du lịch",
                    "đi lại", "ngôn ngữ", "học tập", "tình cảm", "hạnh phúc"
                ]
                
                # Add meanings from our common options that don't conflict
                for meaning in common_wrong_options:
                    if len(filtered_wrong_meanings) >= 3:
                        break
                    if (meaning.lower() not in used_wrong_meanings and 
                        meaning.lower() != question.meaning.lower() and
                        meaning not in filtered_wrong_meanings):
                        filtered_wrong_meanings.append(meaning)
                        used_wrong_meanings.add(meaning.lower())
            
            # Update the question's wrong meanings with our filtered list
            question.wrong_meanings = filtered_wrong_meanings
            
            question.id = question_id
            combined_questions.append(question)
            question_id += 1
    
    combined_title = f"Quiz tổng hợp từ {len(image_files)} ảnh: " + ", ".join(titles[:3])
    if len(titles) > 3:
        combined_title += f" và {len(titles) - 3} ảnh khác"
    
    # Final check to ensure all questions have at least 3 wrong meanings
    for question in combined_questions:
        if question.type == "chinese_to_pinyin_meaning" and len(question.wrong_meanings) < 3:
            # Add some generic wrong answers if needed
            generic_wrong = ["từ khác", "nghĩa khác", "không có nghĩa này", "nghĩa sai"]
            for wrong in generic_wrong:
                if len(question.wrong_meanings) >= 3:
                    break
                if wrong.lower() not in [m.lower() for m in question.wrong_meanings] and wrong.lower() != question.meaning.lower():
                    question.wrong_meanings.append(wrong)
    
    return QuizData(questions=combined_questions, title=combined_title)

def generate_quiz_from_image(image_file):
    """Generate quiz from single image (kept for backward compatibility)"""
    return generate_quiz_from_images([image_file])

def display_question(question: QuizQuestion, question_num: int) -> bool:
    """Display a question based on its type"""
    st.subheader(f"Câu hỏi {question_num}")
    st.write(f"**Loại:** {question.type}")
    
    correct = False
    
    # Handle different question types
    if question.type == "chinese_to_pinyin_meaning":
        correct = display_pinyin_meaning_question(question, question_num)
    elif question.type == "gap_filling":
        correct = display_gap_filling_question(question, question_num)
    elif question.type == "dialogue_arrangement":
        correct = display_dialogue_arrangement_question(question, question_num)
    elif question.type == "reading_comprehension":
        correct = display_reading_comprehension_question(question, question_num)
    else:
        # Default to pinyin and meaning question type
        correct = display_pinyin_meaning_question(question, question_num)
    
    st.write("---")
    return correct

def display_pinyin_meaning_question(question: QuizQuestion, question_num: int) -> bool:
    """Display a question showing Chinese word and asking for pinyin + meaning"""
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
            # Make sure we have at least 4 options (1 correct + 3 wrong)
            if len(question.wrong_meanings) < 3:
                # Add backup options if needed
                backup_options = ["học sinh", "giáo viên", "bạn bè", "gia đình", "thời gian", "từ khác"]
                for option in backup_options:
                    if len(question.wrong_meanings) >= 3:
                        break
                    if option.lower() not in [m.lower() for m in question.wrong_meanings] and option.lower() != question.meaning.lower():
                        question.wrong_meanings.append(option)
            
            # Create and shuffle options
            options = [question.meaning] + question.wrong_meanings[:3]
            random.shuffle(options)
            st.session_state[f"options_{question.id}"] = options
        
        meaning_options = st.session_state[f"options_{question.id}"]
        user_meaning = st.radio(
            "Nghĩa:",
            meaning_options,
            key=f"meaning_{question.id}",
            label_visibility="collapsed"
        )
        
        # For debugging, show the number of options
        st.caption(f"Số lựa chọn: {len(meaning_options)}")
    
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
    
    return correct

def display_gap_filling_question(question: QuizQuestion, question_num: int) -> bool:
    """Display a gap filling question with options"""
    st.markdown(f"### 🇨🇳 Điền từ vào chỗ trống (HSK {question.hsk_level})")
    
    # Display the context sentence with a gap
    context_with_highlight = question.context_sentence.replace("___", "**___**")
    st.write(f"**Câu:** {context_with_highlight}")
    
    # Display options
    st.write("**Chọn từ phù hợp để điền vào chỗ trống:**")
    
    # Get shuffled options for this question instance
    if f"gap_options_{question.id}" not in st.session_state:
        options_copy = question.options.copy()
        random.shuffle(options_copy)
        st.session_state[f"gap_options_{question.id}"] = options_copy
    
    user_answer = st.radio(
        "Lựa chọn:",
        st.session_state[f"gap_options_{question.id}"],
        key=f"gap_{question.id}",
        label_visibility="collapsed"
    )
    
    correct = False
    
    if st.button(f"Kiểm tra câu {question_num}", key=f"check_gap_{question.id}"):
        if user_answer == question.correct_answer:
            st.success("✅ Đúng rồi!")
            correct = True
        else:
            st.error("❌ Sai rồi!")
        
        # Show the complete answer
        st.info("📖 **Đáp án đầy đủ:**")
        filled_sentence = question.context_sentence.replace("___", f"**{question.correct_answer}**")
        st.write(f"Câu đầy đủ: {filled_sentence}")
        
        if question.explanation:
            st.info(f"💡 **Giải thích:** {question.explanation}")
            
        # Show info about the correct word
        st.write(f"**Từ đúng:** {question.correct_answer}")
        st.write(f"**Pinyin:** {question.pinyin}")
        st.write(f"**Nghĩa:** {question.meaning}")
    
    return correct

def display_dialogue_arrangement_question(question: QuizQuestion, question_num: int) -> bool:
    """Display a dialogue arrangement question"""
    st.markdown(f"### 🇨🇳 Sắp xếp hội thoại theo thứ tự đúng")
    st.write(f"**{question.question}**")
    
    # Prepare dialogue parts for arrangement
    if f"dialogue_parts_{question.id}" not in st.session_state:
        dialogue_parts_with_index = [(i, part) for i, part in enumerate(question.dialogue_parts)]
        random.shuffle(dialogue_parts_with_index)
        st.session_state[f"dialogue_parts_{question.id}"] = dialogue_parts_with_index
        st.session_state[f"dialogue_order_{question.id}"] = []
    
    # Display current order
    st.write("**Đã sắp xếp:**")
    order_cols = st.columns(len(st.session_state[f"dialogue_order_{question.id}"]) + 1)
    
    for i, idx in enumerate(st.session_state[f"dialogue_order_{question.id}"]):
        with order_cols[i]:
            st.text_area(
                f"Phần {i+1}",
                question.dialogue_parts[idx],
                height=100,
                key=f"arranged_{question.id}_{i}",
                disabled=True
            )
    
    # Display remaining parts
    st.write("**Các phần còn lại:**")
    remaining_parts = [p for p in st.session_state[f"dialogue_parts_{question.id}"] 
                      if p[0] not in st.session_state[f"dialogue_order_{question.id}"]]
    
    if remaining_parts:
        remaining_cols = st.columns(min(3, len(remaining_parts)))
        for i, (idx, part) in enumerate(remaining_parts):
            with remaining_cols[i % len(remaining_cols)]:
                st.text_area(
                    f"Phần {i+1}",
                    part,
                    height=100,
                    key=f"remaining_{question.id}_{i}",
                    disabled=True
                )
                if st.button(f"Thêm phần này", key=f"add_{question.id}_{i}"):
                    st.session_state[f"dialogue_order_{question.id}"].append(idx)
                    st.rerun()
    
    # Reset button
    if st.button("Reset", key=f"reset_{question.id}"):
        st.session_state[f"dialogue_order_{question.id}"] = []
        st.rerun()
    
    correct = False
    
    if st.button(f"Kiểm tra câu {question_num}", key=f"check_dialogue_{question.id}"):
        user_order = st.session_state[f"dialogue_order_{question.id}"]
        if user_order == question.correct_order:
            st.success("✅ Đúng rồi! Bạn đã sắp xếp đúng thứ tự!")
            correct = True
        else:
            st.error("❌ Sai rồi! Thứ tự đúng là:")
            # Display correct order
            correct_cols = st.columns(len(question.correct_order))
            for i, idx in enumerate(question.correct_order):
                with correct_cols[i]:
                    st.text_area(
                        f"Phần {i+1} (đúng)",
                        question.dialogue_parts[idx],
                        height=100,
                        key=f"correct_{question.id}_{i}",
                        disabled=True
                    )
        
        if question.explanation:
            st.info(f"💡 **Giải thích:** {question.explanation}")
    
    return correct

def display_reading_comprehension_question(question: QuizQuestion, question_num: int) -> bool:
    """Display a reading comprehension question with subquestions"""
    st.markdown(f"### 🇨🇳 阅读理解 (Đọc hiểu)")
    
    # Display the reading text in a larger container
    with st.expander("点击展开阅读文章 (Nhấn để xem bài đọc)", expanded=True):
        st.markdown(f"""
        <div style="background-color: #f5f5f5; padding: 20px; border-radius: 5px; border-left: 5px solid #1E88E5;">
        {question.reading_text}
        </div>
        """, unsafe_allow_html=True)
    
    # Display subquestions (in Chinese)
    all_correct = True
    user_answers = []
    
    for i, subq in enumerate(question.subquestions):
        st.write(f"**问题 {i+1}:** {subq}")
        
        # Get options for this subquestion
        options = question.suboptions[i] if i < len(question.suboptions) else []
        
        user_answer = st.radio(
            f"Câu {i+1}",
            options,
            key=f"subq_{question.id}_{i}",
            label_visibility="collapsed"
        )
        
        user_answers.append(user_answer)
    
    if st.button(f"检查答案 (Kiểm tra)", key=f"check_reading_{question.id}"):
        st.write("### 结果 (Kết quả):")
        
        for i, (user_ans, correct_ans) in enumerate(zip(user_answers, question.subanswers)):
            if user_ans == correct_ans:
                st.success(f"问题 {i+1}: ✅ 正确! (Đúng!)")
            else:
                st.error(f"问题 {i+1}: ❌ 错误! 正确答案是: {correct_ans} (Sai! Đáp án đúng là: {correct_ans})")
                all_correct = False
        
        if all_correct:
            st.success("🎉 恭喜你! 你已经正确回答了所有问题! (Chúc mừng! Bạn đã trả lời đúng tất cả các câu hỏi!)")
        else:
            st.warning("你还没有正确回答所有问题。请再试一次! (Bạn chưa trả lời đúng tất cả các câu hỏi. Hãy thử lại!)")
        
        if question.explanation:
            st.info(f"💡 **解释 (Giải thích):** {question.explanation}")
    
    return all_correct

def main():
    st.set_page_config(
        page_title="Chopchop hoc tieng Trung di",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Chopchop hoc tieng Trung di Quiz Generator")
    st.write("Tải lên một hoặc nhiều hình ảnh ghi chú tiếng Trung để tạo quiz tương tác!")
    st.write("**Format:** Hiển thị từ Hán → Học sinh điền pinyin + chọn nghĩa")
    
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
                    st.image(image, caption=f"Ảnh {i+1}: {uploaded_file.name}", use_container_width=True)
                    image_files.append(uploaded_file)
                st.write(f"**Tổng cộng: {len(uploaded_files)} ảnh**")
            elif use_default:
                try:
                    with open("A.jpg", "rb") as f:
                        image = Image.open("A.jpg")
                        st.image(image, caption="Hình ảnh mẫu (A.jpg)", use_container_width=True)
                        # Reset file pointer for processing
                        image_files = [open("A.jpg", "rb")]
                except FileNotFoundError:
                    st.error("Không tìm thấy file A.jpg!")
                    return
        
        with col2:
            # Generate quiz button
            if st.button("🎯 Tạo Quiz", type="primary"):
                # Show progress bar and status
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner("Đang tạo quiz... Vui lòng đợi!"):
                    try:
                        num_images = len(uploaded_files) if uploaded_files else 1
                        
                        start_time = time.time()
                        
                        # Reset file pointers if using uploaded files
                        if uploaded_files:
                            for uploaded_file in uploaded_files:
                                uploaded_file.seek(0)
                            quiz_data = generate_quiz_from_images(uploaded_files)
                        else:
                            quiz_data = generate_quiz_from_images(image_files)
                        
                        end_time = time.time()
                        processing_time = end_time - start_time
                        
                        progress_bar.progress(1.0)
                        
                        if quiz_data is None:
                            st.error("Không thể tạo quiz từ các ảnh đã tải!")
                            return
                        
                        st.session_state.quiz_data = quiz_data
                        st.session_state.current_question = 0
                        st.session_state.score = 0
                        st.session_state.answered_questions = set()
                        
                        # Clear any previous question options
                        for key in list(st.session_state.keys()):
                            if key.startswith("options_"):
                                del st.session_state[key]
                        
                        status_text.empty()
                        progress_bar.empty()
                        
                        st.success(f"✅ Quiz đã được tạo từ {num_images} ảnh với tổng cộng {len(quiz_data.questions)} câu hỏi!")
                        
                    except Exception as e:
                        progress_bar.empty()
                        status_text.empty()
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
                
                # Type-specific fields
                if question.type == "chinese_to_pinyin_meaning":
                    quiz_text += f"Wrong Options: {', '.join(question.wrong_meanings)}\n"
                
                elif question.type == "gap_filling":
                    quiz_text += f"Context Sentence: {question.context_sentence}\n"
                    quiz_text += f"Options: {', '.join(question.options)}\n"
                    quiz_text += f"Correct Answer: {question.correct_answer}\n"
                    quiz_text += f"HSK Level: {question.hsk_level}\n"
                
                elif question.type == "dialogue_arrangement":
                    quiz_text += "Dialogue Parts:\n"
                    for j, part in enumerate(question.dialogue_parts):
                        quiz_text += f"  Part {j+1}: {part}\n"
                    quiz_text += f"Correct Order: {question.correct_order}\n"
                
                elif question.type == "reading_comprehension":
                    quiz_text += "Reading Text:\n"
                    quiz_text += f"{question.reading_text}\n\n"
                    quiz_text += "Subquestions:\n"
                    for j, (subq, ans) in enumerate(zip(question.subquestions, question.subanswers)):
                        quiz_text += f"  {j+1}. {subq}\n"
                        quiz_text += f"     Answer: {ans}\n"
                        if j < len(question.suboptions):
                            quiz_text += f"     Options: {', '.join(question.suboptions[j])}\n"
                
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
        st.info("Chopchop hoc tieng trung di")

if __name__ == "__main__":
    main()
