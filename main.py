import streamlit as st
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF
from io import BytesIO
import base64
import re
from bs4 import BeautifulSoup
import uuid
import google.generativeai as genai
import os

# Configure Gemini API
GEMINI_API_KEY = "GEMINI_API_KEY"  # Replace with your Gemini API key
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro")

# Set page config
st.set_page_config(
    page_title="Student Performance Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2166ac;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2166ac;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #2166ac;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f5ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2166ac;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = {}

def sanitize_text(text):
    """Sanitize text for PDF generation"""
    if not text:
        return ""
    replacements = {
        "\u2019": "'", "\u2018": "'", "\u201C": '"', "\u201D": '"',
        "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u2022": "*",
        "\u00b7": "-", "\u2010": "-", "\u2011": "-"
    }
    sanitized = text
    for unicode_char, ascii_char in replacements.items():
        sanitized = sanitized.replace(unicode_char, ascii_char)
    try:
        sanitized.encode("latin-1")
        return sanitized
    except UnicodeEncodeError:
        result = sanitized.encode("latin-1", errors="ignore").decode("latin-1")
        return result

def parse_json_data(json_data):
    """Parse uploaded JSON data and extract relevant information"""
    try:
        if isinstance(json_data, list):
            data = json_data[0]
        else:
            data = json_data
        
        # Extract test details
        test_info = data.get("test", {})
        test_name = test_info.get("title", "QPT 1")
        test_date = "11 May 2025"
        total_questions = test_info.get("totalQuestions", 75)
        total_marks = test_info.get("totalMarks", 300)
        test_duration = test_info.get("duration", 3600)

        # Extract subject-wise mapping and performance
        subject_map = {
            "607018ee404ae53194e73d92": "Physics",
            "607018ee404ae53194e73d90": "Chemistry",
            "607018ee404ae53194e73d91": "Mathematics"
        }
        
        subjects = data.get("subjects", [])
        subject_data = pd.DataFrame([
            {
                "Subject": subject_map.get(subj["subjectId"].get("$oid", "Unknown"), "Unknown"),
                "TotalCorrect": subj.get("totalCorrect", 0),
                "TotalAttempted": subj.get("totalAttempted", 0),
                "Accuracy": subj.get("accuracy", 0.0),
                "TotalTimeTaken": subj.get("totalTimeTaken", 0)
            }
            for subj in subjects
        ])

        # Create questions DataFrame
        questions = []
        for section_idx, section in enumerate(data.get("sections", [])):
            section_name = section.get("title", f"Section {section_idx + 1}")
            for q in section.get("questions", []):
                chapter = q.get("questionId", {}).get("chapters", [{"title": "Unknown"}])[0]["title"]
                level = q.get("questionId", {}).get("level", "Unknown")
                subject_id = q.get("subjectId", {})
                if isinstance(subject_id, dict):
                    subject_id = subject_id.get("$oid", "Unknown")
                else:
                    subject_id = str(subject_id).lower() or "Unknown"
                
                is_correct = False
                if q.get("markedOptions", []):
                    is_correct = q["markedOptions"][0].get("isCorrect", False)
                elif q.get("inputValue", {}).get("value", None) is not None:
                    is_correct = q["inputValue"].get("isCorrect", False)
                
                questions.append({
                    "subject": subject_map.get(subject_id, "Unknown"),
                    "chapter": chapter,
                    "level": level,
                    "isCorrect": is_correct,
                    "timeTaken": int(q.get("timeTaken", 0)),
                    "status": str(q.get("status", "")).lower().strip(),
                    "section": section_name
                })
        
        questions_df = pd.DataFrame(questions)
        
        return {
            'questions_df': questions_df,
            'subject_data': subject_data,
            'test_info': {
                'name': test_name,
                'date': test_date,
                'total_questions': total_questions,
                'total_marks': total_marks,
                'duration': test_duration
            },
            'raw_data': data
        }
    except Exception as e:
        st.error(f"Error parsing JSON data: {str(e)}")
        return None

def generate_all_charts(questions_df):
    """Generate all visualization charts and return as bytes for Streamlit"""
    charts = []
    plt.style.use('default')
    sns.set_palette("husl")
    
    try:
        # 1. Histogram of timeTaken
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data=questions_df, x="timeTaken", bins=30, ax=ax)
        ax.set_title("Distribution of Time Taken per Question")
        ax.set_xlabel("Time Taken (s)")
        ax.set_ylabel("Count")
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        charts.append(("time_taken_histogram.png", "This graph shows how long you spent on each question, with taller bars for longer times. It helps you spot which questions slowed you down so you can practice going faster.", buffer.getvalue()))

        # 2. Countplot of section
        if "section" in questions_df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(y="section", data=questions_df, ax=ax)
            ax.set_title("Questions per Section")
            ax.set_xlabel("Count")
            ax.set_ylabel("Section")
            buffer = BytesIO()
            fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            charts.append(("section_count.png", "This chart counts how many questions were in each test section. It shows which sections had more questions, helping you focus your study.", buffer.getvalue()))

        # 3. Countplot of chapter
        fig, ax = plt.subplots(figsize=(6, 8))
        sns.countplot(y="chapter", data=questions_df, ax=ax)
        ax.set_title("Questions per Chapter")
        ax.set_xlabel("Count")
        ax.set_ylabel("Chapter")
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        charts.append(("chapter_count.png", "This graph shows how many questions came from each chapter. It helps you know which chapters need more study if they had lots of questions.", buffer.getvalue()))

        # 4. Countplot of level
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(y="level", data=questions_df, ax=ax)
        ax.set_title("Questions per Difficulty Level")
        ax.set_xlabel("Count")
        ax.set_ylabel("Level")
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        charts.append(("level_count.png", "This chart counts easy, medium, and hard questions. It shows which difficulty levels you faced most, so you can practice the tough ones.", buffer.getvalue()))

        # 5. Countplot of status
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(y="status", data=questions_df, ax=ax)
        ax.set_title("Questions per Answer Status")
        ax.set_xlabel("Count")
        ax.set_ylabel("Status")
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        charts.append(("status_count.png", "This graph shows how many questions you got right, wrong, or skipped. Lots of skipped questions mean you might need to manage time better.", buffer.getvalue()))

        # 6. Lineplot timeTaken vs index
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.lineplot(data=questions_df.reset_index(), x="index", y="timeTaken", ax=ax)
        ax.set_title("Time Taken per Question Over Time")
        ax.set_xlabel("Question Index")
        ax.set_ylabel("Time Taken (s)")
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        charts.append(("time_taken_index.png", "This line shows how long each question took as you went through the test. If the line goes up, later questions took longer, suggesting tiredness or difficulty.", buffer.getvalue()))

        # 7. Lineplot timeTaken by chapter
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.lineplot(data=questions_df, x=questions_df.index, y="timeTaken", hue="chapter", legend=False, ax=ax)
        ax.set_title("Time Taken by Chapter")
        ax.set_xlabel("Question Index")
        ax.set_ylabel("Time Taken (s)")
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        charts.append(("time_taken_chapter.png", "This graph shows time spent on questions from each chapter. High lines mean those chapters took longer, so practice them to get faster.", buffer.getvalue()))

        # 8. Lineplot timeTaken by level
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.lineplot(data=questions_df, x=questions_df.index, y="timeTaken", hue="level", legend=False, ax=ax)
        ax.set_title("Time Taken by Difficulty Level")
        ax.set_xlabel("Question Index")
        ax.set_ylabel("Time Taken (s)")
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        charts.append(("time_taken_level.png", "This graph shows time spent on easy, medium, and hard questions. If hard questions have high lines, practice them to speed up.", buffer.getvalue()))

        # 9. Lineplot timeTaken by section
        if "section" in questions_df.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.lineplot(data=questions_df, x=questions_df.index, y="timeTaken", hue="section", legend=False, ax=ax)
            ax.set_title("Time Taken by Section")
            ax.set_xlabel("Question Index")
            ax.set_ylabel("Time Taken (s)")
            buffer = BytesIO()
            fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            charts.append(("time_taken_section.png", "This graph shows time spent on each test section. High lines mean you were slower in those sections, so practice to improve pacing.", buffer.getvalue()))

        # 10. Heatmap: section vs chapter
        if "section" in questions_df.columns:
            heatmap_df1 = questions_df.pivot_table(index="section", columns="chapter", values="isCorrect", aggfunc="count", fill_value=0)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(heatmap_df1, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
            ax.set_title("Section vs Chapter (Question Count)")
            ax.set_ylabel("Section")
            ax.set_xlabel("Chapter")
            buffer = BytesIO()
            fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            charts.append(("section_vs_chapter_heatmap.png", "This grid shows how many questions each section had from each chapter. Darker boxes mean more questions, guiding your study focus.", buffer.getvalue()))

        # 11. Heatmap: chapter vs level
        heatmap_df2 = questions_df.pivot_table(index="chapter", columns="level", values="isCorrect", aggfunc="count", fill_value=0)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(heatmap_df2, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
        ax.set_title("Chapter vs Level (Question Count)")
        ax.set_ylabel("Chapter")
        ax.set_xlabel("Level")
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        charts.append(("chapter_vs_level_heatmap.png", "This grid shows how many easy, medium, or hard questions each chapter had. Darker boxes highlight chapters with tough questions to practice.", buffer.getvalue()))

        # 12. Heatmap: level vs status
        heatmap_df3 = questions_df.pivot_table(index="level", columns="status", values="isCorrect", aggfunc="count", fill_value=0)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(heatmap_df3, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
        ax.set_title("Level vs Status (Question Count)")
        ax.set_ylabel("Level")
        ax.set_xlabel("Status")
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        charts.append(("level_vs_status_heatmap.png", "This grid shows if easy, medium, or hard questions were right, wrong, or skipped. Darker boxes for wrong answers show where to improve.", buffer.getvalue()))

        # 13. Heatmap: status vs isCorrect
        heatmap_df4 = questions_df.pivot_table(index="status", columns="isCorrect", values="timeTaken", aggfunc="count", fill_value=0)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(heatmap_df4, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
        ax.set_title("Status vs Correctness (Question Count)")
        ax.set_ylabel("Status")
        ax.set_xlabel("Correct")
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        charts.append(("status_vs_correctness_heatmap.png", "This grid shows if answered questions were correct or incorrect. Darker boxes for incorrect answers highlight areas to review.", buffer.getvalue()))

        # 14. Violinplot: section vs timeTaken
        if "section" in questions_df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.violinplot(data=questions_df, y="section", x="timeTaken", scale="width", ax=ax)
            ax.set_title("Time Taken Distribution by Section")
            ax.set_xlabel("Time Taken (s)")
            ax.set_ylabel("Section")
            buffer = BytesIO()
            fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            charts.append(("section_vs_timeTaken_violin.png", "This chart shows time spent on questions in each section, with wider shapes for varied times. It helps you see where your pacing was uneven.", buffer.getvalue()))

        # 15. Violinplot: chapter vs timeTaken
        fig, ax = plt.subplots(figsize=(8, 10))
        sns.violinplot(data=questions_df, y="chapter", x="timeTaken", scale="width", ax=ax)
        ax.set_title("Time Taken Distribution by Chapter")
        ax.set_xlabel("Time Taken (s)")
        ax.set_ylabel("Chapter")
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        charts.append(("chapter_vs_timeTaken_violin.png", "This chart shows time spent on questions from each chapter, with wider shapes for varied times. It highlights chapters where you were slower.", buffer.getvalue()))

        # 16. Violinplot: level vs timeTaken
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.violinplot(data=questions_df, y="level", x="timeTaken", scale="width", ax=ax)
        ax.set_title("Time Taken Distribution by Level")
        ax.set_xlabel("Time Taken (s)")
        ax.set_ylabel("Level")
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        charts.append(("level_vs_timeTaken_violin.png", "This chart shows time spent on easy, medium, and hard questions, with wider shapes for varied times. It shows which difficulty levels slowed you down.", buffer.getvalue()))

        # 17. Violinplot: status vs timeTaken
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.violinplot(data=questions_df, y="status", x="timeTaken", scale="width", ax=ax)
        ax.set_title("Time Taken Distribution by Status")
        ax.set_xlabel("Time Taken (s)")
        ax.set_ylabel("Status")
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        charts.append(("status_vs_timeTaken_violin.png", "This chart shows time spent on correct, incorrect, or skipped questions, with wider shapes for varied times. It highlights if wrong answers took too long.", buffer.getvalue()))

    except Exception as e:
        st.error(f"Error generating charts: {str(e)}")
    
    return charts

def get_gemini_chapters(json_data):
    """Extract chapters with subject bifurcation using Gemini or fallback"""
    syllabus = json_data.get("test", {}).get("syllabus", "")
    sections = json_data.get("sections", [])
    subject_map = {
        "607018ee404ae53194e73d92": "Physics",
        "607018ee404ae53194e73d90": "Chemistry",
        "607018ee404ae53194e73d91": "Mathematics"
    }
    prompt = f"""
You are an expert in processing educational JSON data for test performance analysis. I have a JSON file with:
- A "test.syllabus" field with HTML content listing chapters: {syllabus[:1000]}...
- A "sections" array with questions, each having a "subjectId" (e.g., {list(subject_map.keys())}) and "chapters" (e.g., [{{"title": "Functions"}}]).
- Subject IDs map to: {json.dumps(subject_map)}.

**Task**:
Extract all unique chapter titles and associate them with their subject (Physics, Chemistry, Mathematics) by:
1. Parsing the "test.syllabus" HTML to identify chapter titles and their subjects.
2. Using "sections.questions.chapters" and "subjectId" to associate chapters with subjects.
3. If subject is unclear, infer it from chapter titles (e.g., "Mechanics" → Physics, "Organic Chemistry" → Chemistry, "Functions" → Mathematics).
4. Ensure no duplicate chapters and sort alphabetically within each subject.
5. For Mathematics, only include chapters clearly related to mathematical topics (e.g., Functions, Algebra, Calculus), excluding any Physics or Chemistry chapters (e.g., Electrochemistry, Capacitance).

**Output** (JSON):
```json
{{
  "Physics": ["Chapter 1", "Chapter 2", ...],
  "Chemistry": ["Chapter 1", "Chapter 2", ...],
  "Mathematics": ["Chapter 1", "Chapter 2", ...]
}}
```
"""
    try:
        response = model.generate_content(prompt)
        # Check if response.text is valid JSON
        try:
            chapter_dict = json.loads(response.text.strip("```json\n").strip("\n```"))
        except json.JSONDecodeError as e:
            st.warning(f"Gemini returned invalid JSON: {str(e)}. Falling back to manual chapter extraction.")
            chapter_dict = {"Physics": [], "Chemistry": [], "Mathematics": []}
        for subject in ["Physics", "Chemistry", "Mathematics"]:
            if subject not in chapter_dict:
                chapter_dict[subject] = []
            chapter_dict[subject] = sorted(list(set(chapter_dict[subject])))
    except Exception as e:
        st.warning(f"Gemini chapter extraction failed: {str(e)}. Falling back to manual chapter extraction.")
        chapter_dict = {"Physics": [], "Chemistry": [], "Mathematics": []}
    
    # Fallback chapter extraction
    math_keywords = ["functions", "algebra", "calculus", "geometry", "trigonometry", "sets", "relations", "probability", "statistics"]
    physics_keywords = ["mechanics", "electrostatics", "capacitance", "physics", "force", "energy"]
    chemistry_keywords = ["electrochemistry", "solutions", "organic", "inorganic", "chemistry"]
    if syllabus:
        soup = BeautifulSoup(syllabus, 'html.parser')
        for li in soup.find_all('li'):
            chapter = li.text.strip()
            if chapter:
                chapter_lower = chapter.lower()
                if any(kw in chapter_lower for kw in math_keywords):
                    chapter_dict["Mathematics"].append(chapter)
                elif any(kw in chapter_lower for kw in physics_keywords):
                    chapter_dict["Physics"].append(chapter)
                elif any(kw in chapter_lower for kw in chemistry_keywords):
                    chapter_dict["Chemistry"].append(chapter)
                else:
                    chapter_dict["Mathematics"].append(chapter)
    for section in sections:
        for question in section.get("questions", []):
            subject_id = question.get("subjectId", {})
            if isinstance(subject_id, dict):
                subject_id = subject_id.get("$oid", "Unknown")
            subject = subject_map.get(subject_id, "Mathematics")
            for chapter in question.get("questionId", {}).get("chapters", []):
                chapter_title = chapter.get("title", "").strip()
                if not chapter_title:
                    continue
                chapter_lower = chapter_title.lower()
                if subject == "Mathematics" and any(kw in chapter_lower for kw in physics_keywords + chemistry_keywords):
                    continue  # Skip Physics/Chemistry chapters in Mathematics
                if chapter_title not in chapter_dict[subject]:
                    chapter_dict[subject].append(chapter_title)
    for subject in chapter_dict:
        chapter_dict[subject] = sorted(list(set(chapter_dict[subject])))
    return chapter_dict

def generate_feedback(questions_df, subject_data, chapter_dict, test_info, student_name="Student"):
    """Generate personalized feedback using Gemini or fallback"""
    total = len(questions_df)
    correct = questions_df["isCorrect"].sum()
    accuracy = (correct / total) * 100 if total > 0 else 0
    avg_time = questions_df["timeTaken"].mean()
    time_used = questions_df["timeTaken"].sum() / test_info['duration'] * 100

    subject_summary = "\n".join([
        f"- {s['Subject']}: {s.get('TotalCorrect', 0)}/{s.get('TotalAttempted', 0)} correct ({s.get('Accuracy', 0.0):.2f}%), {s.get('TotalTimeTaken', 0):.1f}s"
        for _, s in subject_data.iterrows()
    ])

    chapter_summary = questions_df.groupby(["subject", "chapter"]).agg({
        "isCorrect": ["count", "mean"],
        "timeTaken": "mean"
    }).reset_index()
    chapter_summary.columns = ["subject", "chapter", "total_questions", "accuracy", "avg_time"]
    chapter_summary["accuracy"] *= 100
    chapter_summary_text = "\n".join([
        f"- {row['subject']} - {row['chapter']}: {row['total_questions']} questions, {row['accuracy']:.2f}% accuracy, {row['avg_time']:.2f}s avg time"
        for _, row in chapter_summary.iterrows()
    ])

    weakest_chapter = chapter_summary.loc[chapter_summary["accuracy"].idxmin()] if not chapter_summary.empty else pd.Series({
        "chapter": "N/A", "accuracy": 0, "subject": "N/A"
    })
    strongest_chapter = chapter_summary.loc[chapter_summary["accuracy"].idxmax()] if not chapter_summary.empty else pd.Series({
        "chapter": "N/A", "accuracy": 0, "subject": "N/A"
    })

    difficulty_accuracy = questions_df.groupby("level")["isCorrect"].mean() * 100
    toughest_level = difficulty_accuracy.idxmin() if difficulty_accuracy.size else "N/A"
    toughest_level_acc = difficulty_accuracy.min() if difficulty_accuracy.size else 0

    slow_questions = questions_df[questions_df["timeTaken"] > questions_df["timeTaken"].quantile(0.75)]
    slow_acc = slow_questions["isCorrect"].mean() * 100 if not slow_questions.empty else 0

    prompt = f"""
You are an expert educational assistant creating a personalized feedback report for {student_name} based on their performance in {test_info['name']} ({test_info['date']}). Use the provided data to craft a motivating, data-driven narrative with highly specific, chapter-focused actionable suggestions. Avoid generic advice.

**Performance Data**:
- Total Questions: {total}
- Correct Answers: {correct}
- Total Marks Scored: {correct * (test_info['total_marks'] / test_info['total_questions']):.2f}/{test_info['total_marks']}
- Accuracy: {accuracy:.2f}%
- Average Time per Question: {avg_time:.2f}s
- Time Used: {time_used:.2f}% of {test_info['duration']}s
- Weakest Chapter: {weakest_chapter['chapter']} in {weakest_chapter['subject']} ({weakest_chapter['accuracy']:.2f}%)
- Strongest Chapter: {strongest_chapter['chapter']} in {strongest_chapter['subject']} ({strongest_chapter['accuracy']:.2f}%)
- Toughest Difficulty: {toughest_level} ({toughest_level_acc:.2f}%)
- Accuracy on Slow Questions: {slow_acc:.2f}%
- Subject-wise Performance:
{subject_summary}
- Chapter-wise Performance:
{chapter_summary_text}
- Chapters by Subject:
{json.dumps(chapter_dict, indent=2)}

**Instructions**:
- **Intro (100–150 words)**: Greet {student_name}, acknowledge effort, highlight strengths (e.g., strongest chapter), and encourage improvement in weaker chapters.
- **Performance Breakdown (200–300 words)**:
  - **Subject-wise**: Summarize performance per subject (accuracy, time), noting strongest and weakest subjects.
  - **Chapter-wise**: Analyze performance by chapter, focusing on weakest and strongest chapters per subject, and identify patterns (e.g., low accuracy or high time).
  - **Difficulty-wise**: Evaluate accuracy and time across difficulty levels (easy, medium, hard).
  - **Time vs. Accuracy**: Analyze time spent vs. accuracy, noting trends (e.g., slower questions with lower accuracy).
  - **Overall Metrics**: Summarize marks, time utilization, and accuracy.
- **Actionable Suggestions (200–250 words)**: Generate 4–5 specific, data-driven suggestions per subject (Physics, Chemistry, Mathematics), focusing on chapters:
  - Each suggestion must:
    - Start with '-'.
    - Reference specific chapters, accuracy, or time metrics from the data.
    - For Mathematics, only suggest improvements for chapters listed in the Mathematics section of Chapters by Subject (e.g., Functions, Sets and Relations, excluding Electrochemistry, Capacitance, etc.).
    - Be practical, tailored, and avoid generic advice (e.g., instead of "study more," suggest "practice Functions problems to improve 38.89% accuracy").
- **Tone**: Friendly, encouraging, specific, motivating.
- **Output Format** (markdown):
```markdown
### Intro
...
### Performance Breakdown
#### Subject-wise Analysis
...
#### Chapter-wise Analysis
...
#### Difficulty-wise Analysis
...
#### Time and Accuracy Insights
...
#### Overall Metrics
...
### Actionable Suggestions
**Physics:**
- ...
**Chemistry:**
- ...
**Mathematics:**
- ...
```
"""
    try:
        response = model.generate_content(prompt)
        feedback_text = response.text.strip()
        if not feedback_text:
            raise ValueError("Gemini returned empty feedback")
        return sanitize_text(feedback_text)
    except Exception as e:
        st.error(f"Gemini API Error: {str(e)}")
        suggestions = []
        for subject in ["Physics", "Chemistry", "Mathematics"]:
            subject_chapters = chapter_summary[chapter_summary["subject"] == subject]
            if subject_chapters.empty:
                suggestions.append(f"**{subject}:**\n- No chapter data available; practice core topics in {subject} to build confidence.")
                continue
            weakest = subject_chapters.loc[subject_chapters["accuracy"].idxmin()] if not subject_chapters.empty else None
            slowest = subject_chapters.loc[subject_chapters["avg_time"].idxmax()] if not subject_chapters.empty else None
            suggestions.append(f"**{subject}:**")
            if weakest is not None and (subject != "Mathematics" or weakest["chapter"] in chapter_dict["Mathematics"]):
                suggestions.append(f"- Focus on {weakest['chapter']} ({weakest['accuracy']:.2f}% accuracy); practice targeted problems to improve.")
                suggestions.append(f"- Review {weakest['chapter']} concepts, as low accuracy suggests gaps in understanding.")
            if slowest is not None and (subject != "Mathematics" or slowest["chapter"] in chapter_dict["Mathematics"]):
                suggestions.append(f"- Speed up on {slowest['chapter']} (avg {slowest['avg_time']:.2f}s); use timed quizzes to improve pacing.")
            suggestions.append(f"- Revisit {subject} chapters with low accuracy (<60%) using past papers.")
            suggestions.append(f"- Strengthen {subject} by solving mixed-difficulty problems from {subject_chapters['chapter'].iloc[0] if not subject_chapters.empty else 'core topics'}.")
        suggestions_text = "\n".join(suggestions)
        fallback = f"""
### Intro
Dear {student_name}, great effort on {test_info['name']}! Your performance in {strongest_chapter['chapter']} ({strongest_chapter['accuracy']:.2f}%) shines, showing your potential. Areas like {weakest_chapter['chapter']} ({weakest_chapter['accuracy']:.2f}%) offer growth opportunities. Let's dive into your results!

### Performance Breakdown
#### Subject-wise Analysis
{subject_summary}
- Strongest: {subject_data.loc[subject_data['Accuracy'].idxmax(), 'Subject']} ({subject_data['Accuracy'].max():.2f}%).
- Weakest: {subject_data.loc[subject_data['Accuracy'].idxmin(), 'Subject']} ({subject_data['Accuracy'].min():.2f}%).

#### Chapter-wise Analysis
{chapter_summary_text}

#### Difficulty-wise Analysis
- Toughest: {toughest_level} ({toughest_level_acc:.2f}%).

#### Time and Accuracy Insights
- Slow Questions: {slow_acc:.2f}% accuracy.
- Time Used: {time_used:.2f}% ({avg_time:.2f}s/question).

#### Overall Metrics
- Marks: {correct * (test_info['total_marks'] / test_info['total_questions']):.2f}/{test_info['total_marks']}
- Accuracy: {accuracy:.2f}%

### Actionable Suggestions
{suggestions_text}
"""
        return sanitize_text(fallback)

class PDF(FPDF):
    def __init__(self, logo_path=None):
        super().__init__()
        self.logo_path = logo_path

    def header(self):
        if self.page_no() > 1:
            if self.logo_path and os.path.exists(self.logo_path):
                self.image(self.logo_path, x=170, y=8, w=25)
            self.set_line_width(0.3)
            self.set_draw_color(33, 102, 172)
            self.line(10, 20, 200, 20)
            self.ln(15)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no() - 1}", align="C", ln=1)

    def section_title(self, title):
        self.set_fill_color(240, 245, 255)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(33, 102, 172)
        self.cell(0, 8, title, align="L", fill=True, ln=1)
        self.ln(4)

    def plain_section_title(self, title):
        self.set_font("Helvetica", "", 14)
        self.set_text_color(40, 40, 40)
        self.cell(0, 8, title, align="L", ln=1)
        self.ln(6)

    def subtitle(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(60, 60, 120)
        self.cell(0, 6, title, align="L", ln=1)
        self.ln(2)

    def add_table(self, data, headers, widths, headers_map, title=""):
        if title:
            self.subtitle(title)
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(230, 230, 230)
        self.set_draw_color(160, 160, 160)
        for header in headers:
            self.cell(widths[headers.index(header)], 8, header, border=1, align="C", fill=True)
        self.ln()
        self.set_font("Helvetica", "", 8)
        for i, row in data.iterrows():
            fill_color = (255, 255, 255) if i % 2 == 0 else (250, 250, 250)
            self.set_fill_color(*fill_color)
            for header, width in zip(headers, widths):
                col_key = headers_map.get(header, header)
                if col_key not in data.columns:
                    value = "N/A"
                else:
                    value = row[col_key]
                    value = f"{value:.2f}" if isinstance(value, (float, np.floating)) else str(value)
                self.cell(width, 7, value, border=1, align="C", fill=True)
            self.ln()
        self.ln(6)

    def add_image(self, image_data, caption, description):
        try:
            if isinstance(image_data, bytes):
                from io import BytesIO
                img_buffer = BytesIO(image_data)
                self.image(img_buffer, x=30, w=150, h=80, type='PNG')
                self.ln(2)
                self.set_font("Helvetica", "I", 8)
                self.set_text_color(90, 90, 90)
                self.multi_cell(0, 4, caption, align="C")
                self.ln(2)
                self.set_font("Helvetica", "", 9)
                self.set_text_color(40, 40, 40)
                self.multi_cell(0, 5, description, align="L")
            else:
                self.set_font("Helvetica", "", 10)
                self.set_text_color(255, 0, 0)
                self.multi_cell(0, 6, f"[Image Missing: {caption}]")
            self.ln(6)
        except Exception as e:
            st.error(f"Image error: {str(e)}")
            self.set_font("Helvetica", "", 10)
            self.set_text_color(255, 0, 0)
            self.multi_cell(0, 6, f"[Image Error: {str(e)}]")

    def add_suggestions(self, suggestions_text):
        if not suggestions_text.strip():
            self.set_font("Helvetica", "", 10)
            self.set_text_color(255, 0, 0)
            self.multi_cell(0, 6, "[No suggestions provided]")
            return
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        lines = suggestions_text.split("\n")
        current_subject = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("**") and line.endswith("**"):
                current_subject = line.strip("**").rstrip(":")
                self.set_font("Helvetica", "B", 10)
                self.multi_cell(0, 6, current_subject)
                self.set_font("Helvetica", "", 10)
                self.ln(2)
            elif (line.startswith("-") or line.startswith("*")) and current_subject:
                suggestion = line.lstrip("-* ").strip()
                if suggestion:
                    self.multi_cell(0, 6, f"* {suggestion}")
                    self.ln(1)
            elif line and current_subject:
                self.multi_cell(0, 6, f"* {line}")
                self.ln(1)

def generate_analysis_pdf(questions_df, subject_data, feedback_sections, chapter_dict, image_list, test_info, student_name="Student"):
    """Generate a comprehensive PDF report"""
    pdf = PDF(logo_path=None)  # No logo for simplicity in Streamlit
    pdf.set_auto_page_break(auto=True, margin=15)

    # Cover Page
    pdf.add_page()
    pdf.set_font("Arial", "B", 24)
    pdf.set_text_color(33, 102, 172)
    pdf.cell(0, 20, "[Book] Student Performance Report", align="C", ln=1)
    pdf.ln(60)
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 10, f"{test_info['name']} - {test_info['date']}", align="C", ln=1)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"For: {student_name}", align="C", ln=1)
    pdf.cell(0, 10, "Generated by MathonGo AI", align="C", ln=1)

    # Summary Statistics
    pdf.add_page()
    pdf.section_title("1. Summary Statistics")
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(40, 40, 40)
    total = len(questions_df)
    correct = questions_df["isCorrect"].sum()
    accuracy = (correct / total) * 100 if total > 0 else 0
    avg_time = questions_df["timeTaken"].mean()
    marks_scored = correct * (test_info['total_marks'] / test_info['total_questions'])
    time_used = questions_df["timeTaken"].sum() / test_info['duration'] * 100
    stats = [
        f"[Note] Total Questions: {total}",
        f"[Check] Correct Answers: {correct}",
        f"[Target] Accuracy: {accuracy:.2f}%",
        f"[Timer] Average Time per Question: {avg_time:.2f}s",
        f"[Marks] Marks Scored: {marks_scored:.1f}/{test_info['total_marks']}",
        f"[Timer] Time Used: {time_used:.2f}% of {test_info['duration']}s"
    ]
    for stat in stats:
        pdf.multi_cell(0, 6, stat)
        pdf.ln(4)

    # Personalized Feedback
    pdf.section_title("2. Personalized Feedback")
    pdf.set_font("Arial", "", 10)
    intro_text = sanitize_text(feedback_sections["intro"])
    if intro_text.strip():
        pdf.multi_cell(0, 5, intro_text)
    else:
        pdf.multi_cell(0, 5, "[No feedback provided]")
    pdf.ln(6)

    # Performance Breakdown
    pdf.section_title("3. Performance Analysis")
    pdf.set_font("Arial", "", 10)
    for section, key in [
        ("Subject-wise Analysis", "subject_breakdown"),
        ("Chapter-wise Analysis", "chapter_breakdown"),
        ("Difficulty-wise Analysis", "difficulty_breakdown"),
        ("Time vs. Accuracy", "time_breakdown"),
        ("Overall Metrics", "overall_breakdown")
    ]:
        text = sanitize_text(feedback_sections[key])
        if text.strip():
            pdf.subtitle(section)
            pdf.multi_cell(0, 5, text)
            pdf.ln(4)

    # Subject-wise table
    pdf.add_table(
        data=subject_data[["Subject", "TotalCorrect", "TotalAttempted", "Accuracy", "TotalTimeTaken"]],
        headers=["Subject", "Correct", "Attempted", "Accuracy (%)", "Time (s)"],
        widths=[50, 30, 30, 30, 30],
        headers_map={
            "Subject": "Subject",
            "Correct": "TotalCorrect",
            "Attempted": "TotalAttempted",
            "Accuracy (%)": "Accuracy",
            "Time (%)": "TotalTimeTaken"
        },
        title="Subject-wise Performance"
    )

    # Actionable Suggestions
    pdf.plain_section_title("4. Actionable Suggestions")
    pdf.add_suggestions(sanitize_text(feedback_sections["actionable_suggestions"]))

    # Chapter-wise Analysis
    pdf.section_title("5. Chapter-wise Analysis")
    chapter_summary = questions_df.groupby(["chapter"]).agg({
        "isCorrect": ["count", "sum", "mean"],
        "timeTaken": ["mean"]
    }).reset_index()
    chapter_summary.columns = ["Chapter", "Questions", "Correct", "Accuracy (%)", "Avg Time (s)"]
    chapter_summary["Accuracy (%)"] = chapter_summary["Accuracy (%)"] * 100
    chapter_summary = chapter_summary.sort_values(by=["Accuracy (%)"], ascending=[False])

    if not chapter_summary.empty:
        pdf.add_table(
            data=chapter_summary,
            headers=["Chapter", "Questions", "Correct", "Accuracy (%)", "Avg Time (s)"],
            widths=[60, 30, 30, 30, 30],
            headers_map={
                "Chapter": "Chapter",
                "Questions": "Questions",
                "Correct": "Correct",
                "Accuracy (%)": "Accuracy (%)",
                "Avg Time (s)": "Avg Time (s)"
            },
            title="Performance by Chapter"
        )
    else:
        pdf.set_font("Arial", "", 10)
        pdf.set_text_color(255, 0, 0)
        pdf.multi_cell(0, 5, f"[No chapters found in the test data]")
        pdf.ln(4)

    # Visual Insights
    pdf.section_title("6. Visual Insights")
    for i, (image_name, description, image_data) in enumerate(image_list):
        if i % 2 == 0 and i > 0:
            pdf.add_page()
        caption = image_name.replace(".png", "").replace("_", " ").title()
        pdf.add_image(image_data, caption, description)

    # Save to bytes
    pdf_buffer = BytesIO()
    pdf.output(pdf_buffer, dest="S")
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

def generate_summary_stats(questions_df, subject_data, test_info):
    """Generate summary statistics"""
    total_questions = len(questions_df)
    correct_answers = questions_df['isCorrect'].sum()
    accuracy = (correct_answers / total_questions * 100) if total_questions > 0 else 0
    avg_time = questions_df['timeTaken'].mean()
    total_time = questions_df['timeTaken'].sum()
    time_percentage = (total_time / test_info['duration'] * 100) if test_info['duration'] > 0 else 0
    marks_scored = correct_answers * (test_info['total_marks'] / test_info['total_questions'])
    
    return {
        'total_questions': total_questions,
        'correct_answers': correct_answers,
        'accuracy': accuracy,
        'avg_time': avg_time,
        'total_time': total_time,
        'time_percentage': time_percentage,
        'marks_scored': marks_scored,
        'total_marks': test_info['total_marks']
    }

def generate_basic_feedback(questions_df, subject_data, stats):
    """Generate basic feedback without AI"""
    feedback = {}
    
    # Introduction
    feedback['intro'] = f"""
    Great effort on your test! You answered {stats['correct_answers']} out of {stats['total_questions']} questions correctly, 
    achieving an accuracy of {stats['accuracy']:.1f}%. You used {stats['time_percentage']:.1f}% of the total time available.
    """
    
    # Subject analysis
    best_subject = subject_data.loc[subject_data['Accuracy'].idxmax(), 'Subject'] if not subject_data.empty else "N/A"
    worst_subject = subject_data.loc[subject_data['Accuracy'].idxmin(), 'Subject'] if not subject_data.empty else "N/A"
    
    feedback['subject_analysis'] = f"""
    Your strongest subject appears to be {best_subject}, while {worst_subject} shows room for improvement.
    Focus on practicing more problems in your weaker areas.
    """
    
    # Time analysis
    if stats['avg_time'] > 120:  # More than 2 minutes per question
        feedback['time_analysis'] = "You're spending significant time per question. Consider practicing time management strategies."
    else:
        feedback['time_analysis'] = "Your time management seems reasonable. Keep up the good pacing!"
    
    # Chapter analysis
    if 'chapter' in questions_df.columns:
        chapter_accuracy = questions_df.groupby('chapter')['isCorrect'].mean() * 100
        weakest_chapter = chapter_accuracy.idxmin() if not chapter_accuracy.empty else "N/A"
        strongest_chapter = chapter_accuracy.idxmax() if not chapter_accuracy.empty else "N/A"
        
        feedback['chapter_analysis'] = f"""
        Your strongest chapter is {strongest_chapter} ({chapter_accuracy.max():.1f}% accuracy).
        Focus more on {weakest_chapter} ({chapter_accuracy.min():.1f}% accuracy) for improvement.
        """
    else:
        feedback['chapter_analysis'] = "Chapter-wise analysis not available."
    
    return feedback

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Student Performance Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📋 Navigation")
    st.sidebar.markdown("---")
    
    # File uploader
    st.sidebar.subheader("📁 Upload Data")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a JSON file",
        type=['json'],
        help="Upload your student performance JSON file"
    )
    
    # Student name input
    student_name = st.sidebar.text_input("👤 Student Name", value="Student", help="Enter the student's name for the report")
    
    if uploaded_file is not None:
        try:
            # Load and parse data
            json_data = json.load(uploaded_file)
            parsed_data = parse_json_data(json_data)
            
            if parsed_data:
                st.session_state.analysis_data = parsed_data
                st.session_state.data_loaded = True
                st.sidebar.success("✅ Data loaded successfully!")
                
                # Navigation
                analysis_sections = [
                    "📊 Overview",
                    "📈 Visualizations", 
                    "📋 Detailed Analysis",
                    "📄 Generate Report"
                ]
                
                selected_section = st.sidebar.radio("Select Section", analysis_sections)
                
                # Get data
                questions_df = parsed_data['questions_df']
                subject_data = parsed_data['subject_data']
                test_info = parsed_data['test_info']
                
                # Generate stats and feedback
                stats = generate_summary_stats(questions_df, subject_data, test_info)
                feedback = generate_basic_feedback(questions_df, subject_data, stats)
                
                # Main content based on selection
                if selected_section == "📊 Overview":
                    st.markdown('<h2 class="section-header">Test Overview</h2>', unsafe_allow_html=True)
                    
                    # Test Info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>📝 {test_info['name']}</h3>
                            <p><strong>Date:</strong> {test_info['date']}</p>
                            <p><strong>Duration:</strong> {test_info['duration']} seconds</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>📊 Performance</h3>
                            <p><strong>Accuracy:</strong> {stats['accuracy']:.1f}%</p>
                            <p><strong>Score:</strong> {stats['marks_scored']:.1f}/{stats['total_marks']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>⏱️ Time Usage</h3>
                            <p><strong>Avg per Q:</strong> {stats['avg_time']:.1f}s</p>
                            <p><strong>Total Used:</strong> {stats['time_percentage']:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Key Metrics
                    st.markdown('<h3 class="section-header">Key Metrics</h3>', unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Questions", stats['total_questions'])
                    col2.metric("Correct Answers", stats['correct_answers'])
                    col3.metric("Accuracy", f"{stats['accuracy']:.1f}%")
                    col4.metric("Time Efficiency", f"{stats['time_percentage']:.1f}%")
                    
                    # Subject Performance Table
                    if not subject_data.empty:
                        st.markdown('<h3 class="section-header">Subject-wise Performance</h3>', unsafe_allow_html=True)
                        st.dataframe(subject_data, use_container_width=True)
                
                elif selected_section == "📈 Visualizations":
                    st.markdown('<h2 class="section-header">Performance Visualizations</h2>', unsafe_allow_html=True)
                    
                    # Generate charts
                    with st.spinner("Generating visualizations..."):
                        charts = generate_all_charts(questions_df)
                    
                    if charts:
                        # Display charts in tabs
                        chart_tabs = st.tabs([chart[0].replace(".png", "").replace("_", " ").title() for chart in charts])
                        
                        for i, (_, _, image_data) in enumerate(charts):
                            with chart_tabs[i]:
                                st.image(image_data, use_column_width=True)
                    else:
                        st.warning("No visualizations could be generated from the data.")
                
                elif selected_section == "📋 Detailed Analysis":
                    st.markdown('<h2 class="section-header">Detailed Performance Analysis</h2>', unsafe_allow_html=True)
                    
                    # Feedback sections
                    st.subheader("📝 Introduction")
                    st.write(feedback['intro'])
                    
                    st.subheader("📚 Subject Analysis")
                    st.write(feedback['subject_analysis'])
                    
                    st.subheader("⏰ Time Management")
                    st.write(feedback['time_analysis'])
                    
                    st.subheader("📖 Chapter Analysis")
                    st.write(feedback['chapter_analysis'])
                    
                    # Detailed data tables
                    with st.expander("📊 View Detailed Question Data"):
                        st.dataframe(questions_df, use_container_width=True)
                    
                    # Chapter-wise performance if available
                    if 'chapter' in questions_df.columns and questions_df['chapter'].nunique() > 1:
                        with st.expander("📈 Chapter-wise Performance"):
                            chapter_perf = questions_df.groupby('chapter').agg({
                                'isCorrect': ['count', 'sum', 'mean'],
                                'timeTaken': 'mean'
                            }).round(3)
                            chapter_perf.columns = ['Total Questions', 'Correct', 'Accuracy', 'Avg Time']
                            chapter_perf['Accuracy'] = (chapter_perf['Accuracy'] * 100).round(1)
                            st.dataframe(chapter_perf, use_container_width=True)
                
                elif selected_section == "📄 Generate Report":
                    st.markdown('<h2 class="section-header">Generate PDF Report</h2>', unsafe_allow_html=True)
                    
                    st.write("Click the button below to generate a comprehensive PDF report of the student's performance.")
                    
                    if st.button("🔄 Generate PDF Report", type="primary"):
                        with st.spinner("Generating PDF report..."):
                            try:
                                # Generate charts
                                image_list = generate_all_charts(questions_df)
                                # Generate chapter dictionary
                                chapter_dict = get_gemini_chapters(parsed_data['raw_data'])
                                # Generate feedback
                                feedback_raw = generate_feedback(questions_df, subject_data, chapter_dict, test_info, student_name)
                                feedback_sections = {
                                    "intro": "",
                                    "subject_breakdown": "",
                                    "chapter_breakdown": "",
                                    "difficulty_breakdown": "",
                                    "time_breakdown": "",
                                    "overall_breakdown": "",
                                    "suggestions": ""
                                }
                                current_section = ""
                                for line in feedback_raw.split("\n"):
                                    line = line.strip()
                                    if not line:
                                        continue
                                    if line.startswith("### Intro"):
                                        current_section = "intro"
                                    elif line.startswith("### Performance Breakdown"):
                                        current_section = ""
                                    elif line.startswith("#### Subject-wise Analysis"):
                                        current_section = "subject_breakdown"
                                    elif line.startswith("#### Chapter-wise Analysis"):
                                        current_section = "chapter_breakdown"
                                    elif line.startswith("#### Difficulty-wise Analysis"):
                                        current_section = "difficulty_breakdown"
                                    elif line.startswith("#### Time and Accuracy Insights"):
                                        current_section = "time_breakdown"
                                    elif line.startswith("#### Overall Metrics"):
                                        current_section = "overall_breakdown"
                                    elif line.startswith("### Actionable Suggestions"):
                                        current_section = "suggestions"
                                    elif current_section:
                                        feedback_sections[current_section] += line + "\n"
                                
                                # Generate PDF
                                pdf_buffer = generate_analysis_pdf(
                                    questions_df, subject_data, feedback_sections, chapter_dict, image_list, test_info, student_name
                                )
                                
                                st.success("✅ PDF report generated successfully!")
                                
                                # Download button
                                st.download_button(
                                    label="📥 Download PDF Report",
                                    data=pdf_buffer.getvalue(),
                                    file_name=f"{student_name}_Performance_Report.pdf",
                                    mime="application/pdf"
                                )
                                
                            except Exception as e:
                                st.error(f"❌ Error generating PDF: {str(e)}")
            else:
                st.error("❌ Failed to parse the JSON data. Please check the file format.")
                
        except json.JSONDecodeError:
            st.error("❌ Invalid JSON file. Please upload a valid JSON file.")
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to the Student Performance Analysis Dashboard! 👋
        
        This tool helps you analyze student test performance data and generate comprehensive reports.
        
        ### Features:
        - 📊 **Interactive Visualizations**: Charts and graphs showing performance patterns
        - 📈 **Subject-wise Analysis**: Detailed breakdown by subject and chapter  
        - ⏱️ **Time Management Insights**: Analysis of time spent per question
        - 📄 **PDF Report Generation**: Professional reports for students and teachers
        - 🎯 **Performance Metrics**: Accuracy, efficiency, and improvement areas
        
        ### How to Use:
        1. **Upload** your JSON performance data file using the sidebar
        2. **Enter** the student's name for personalized reports
        3. **Explore** different sections using the navigation menu
        4. **Generate** and download PDF reports
        
        ### Supported Data Format:
        - JSON files containing student test performance data
        - Must include question-wise responses, timing, and correctness information
        
        **Get started by uploading a JSON file in the sidebar!** 🚀
        """)
        
        # Sample data structure info
        with st.expander("📋 Expected JSON Data Structure"):
            st.code('''
            {
                "test": {
                    "title": "Test Name",
                    "totalQuestions": 75,
                    "totalMarks": 300,
                    "duration": 3600
                },
                "subjects": [
                    {
                        "subjectId": {"$oid": "subject_id"},
                        "totalCorrect": 20,
                        "totalAttempted": 25,
                        "accuracy": 0.8,
                        "totalTimeTaken": 1800
                    }
                ],
                "sections": [
                    {
                        "title": "Section Name",
                        "questions": [
                            {
                                "questionId": {
                                    "chapters": [{"title": "Chapter Name"}],
                                    "level": "Easy"
                                },
                                "subjectId": "subject_id",
                                "markedOptions": [{"isCorrect": true}],
                                "timeTaken": 120,
                                "status": "answered"
                            }
                        ]
                    }
                ]
            }
            ''', language='json')

if __name__ == "__main__":
    main()