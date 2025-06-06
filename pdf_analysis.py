import json
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF
from google.colab import files
import google.generativeai as genai
from bs4 import BeautifulSoup
import uuid

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyD2PbjMPJ8JgiaPKs6h7d3qMaS8sHu-l7Q"  # Replace with your Gemini API key
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro")

# Load JSON data
json_path = "sample_submission_analysis_3.json"
try:
    with open(json_path, "r", encoding='utf-8') as f:
        data = json.load(f)[0]
except Exception as e:
    print(f"Error loading JSON: {e}")
    data = {}

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

# Set Seaborn style
sns.set(style="whitegrid", palette="muted")

# Generate all charts with explanations
def generate_all_charts(questions_df, save_dir="charts/"):
    charts = []
    try:
        os.makedirs(save_dir, exist_ok=True)

        # 1. Histogram of timeTaken
        plt.figure(figsize=(6, 4))
        sns.histplot(data=questions_df, x="timeTaken", bins=30)
        plt.title("Distribution of Time Taken per Question")
        plt.xlabel("Time Taken (s)")
        plt.ylabel("Count")
        plt_path = os.path.join(save_dir, "time_taken_histogram.png")
        plt.savefig(plt_path, dpi=300, bbox_inches="tight")
        plt.close()
        charts.append((plt_path, "This graph shows how long you spent on each question, with taller bars for longer times. It helps you spot which questions slowed you down so you can practice going faster."))

        # 2. Countplot of section
        if "section" in questions_df.columns:
            plt.figure(figsize=(6, 4))
            sns.countplot(y="section", data=questions_df)
            plt.title("Questions per Section")
            plt.xlabel("Count")
            plt.ylabel("Section")
            plt_path = os.path.join(save_dir, "section_count.png")
            plt.savefig(plt_path, dpi=300, bbox_inches="tight")
            plt.close()
            charts.append((plt_path, "This chart counts how many questions were in each test section. It shows which sections had more questions, helping you focus your study."))

        # 3. Countplot of chapter
        plt.figure(figsize=(6, 8))
        sns.countplot(y="chapter", data=questions_df)
        plt.title("Questions per Chapter")
        plt.xlabel("Count")
        plt.ylabel("Chapter")
        plt_path = os.path.join(save_dir, "chapter_count.png")
        plt.savefig(plt_path, dpi=300, bbox_inches="tight")
        plt.close()
        charts.append((plt_path, "This graph shows how many questions came from each chapter. It helps you know which chapters need more study if they had lots of questions."))

        # 4. Countplot of level
        plt.figure(figsize=(6, 4))
        sns.countplot(y="level", data=questions_df)
        plt.title("Questions per Difficulty Level")
        plt.xlabel("Count")
        plt.ylabel("Level")
        plt_path = os.path.join(save_dir, "level_count.png")
        plt.savefig(plt_path, dpi=300, bbox_inches="tight")
        plt.close()
        charts.append((plt_path, "This chart counts easy, medium, and hard questions. It shows which difficulty levels you faced most, so you can practice the tough ones."))

        # 5. Countplot of status
        plt.figure(figsize=(6, 4))
        sns.countplot(y="status", data=questions_df)
        plt.title("Questions per Answer Status")
        plt.xlabel("Count")
        plt.ylabel("Status")
        plt_path = os.path.join(save_dir, "status_count.png")
        plt.savefig(plt_path, dpi=300, bbox_inches="tight")
        plt.close()
        charts.append((plt_path, "This graph shows how many questions you got right, wrong, or skipped. Lots of skipped questions mean you might need to manage time better."))

        # 6. Lineplot timeTaken vs index
        plt.figure(figsize=(8, 4))
        sns.lineplot(data=questions_df.reset_index(), x="index", y="timeTaken")
        plt.title("Time Taken per Question Over Time")
        plt.xlabel("Question Index")
        plt.ylabel("Time Taken (s)")
        plt_path = os.path.join(save_dir, "time_taken_index.png")
        plt.savefig(plt_path, dpi=300, bbox_inches="tight")
        plt.close()
        charts.append((plt_path, "This line shows how long each question took as you went through the test. If the line goes up, later questions took longer, suggesting tiredness or difficulty."))

        # 7. Lineplot timeTaken by chapter
        plt.figure(figsize=(8, 4))
        sns.lineplot(data=questions_df, x=questions_df.index, y="timeTaken", hue="chapter", legend=False)
        plt.title("Time Taken by Chapter")
        plt.xlabel("Question Index")
        plt.ylabel("Time Taken (s)")
        plt_path = os.path.join(save_dir, "time_taken_chapter.png")
        plt.savefig(plt_path, dpi=300, bbox_inches="tight")
        plt.close()
        charts.append((plt_path, "This graph shows time spent on questions from each chapter. High lines mean those chapters took longer, so practice them to get faster."))

        # 8. Lineplot timeTaken by level
        plt.figure(figsize=(8, 4))
        sns.lineplot(data=questions_df, x=questions_df.index, y="timeTaken", hue="level", legend=False)
        plt.title("Time Taken by Difficulty Level")
        plt.xlabel("Question Index")
        plt.ylabel("Time Taken (s)")
        plt_path = os.path.join(save_dir, "time_taken_level.png")
        plt.savefig(plt_path, dpi=300, bbox_inches="tight")
        plt.close()
        charts.append((plt_path, "This graph shows time spent on easy, medium, and hard questions. If hard questions have high lines, practice them to speed up."))

        # 9. Lineplot timeTaken by section
        if "section" in questions_df.columns:
            plt.figure(figsize=(8, 4))
            sns.lineplot(data=questions_df, x=questions_df.index, y="timeTaken", hue="section", legend=False)
            plt.title("Time Taken by Section")
            plt.xlabel("Question Index")
            plt.ylabel("Time Taken (s)")
            plt_path = os.path.join(save_dir, "time_taken_section.png")
            plt.savefig(plt_path, dpi=300, bbox_inches="tight")
            plt.close()
            charts.append((plt_path, "This graph shows time spent on each test section. High lines mean you were slower in those sections, so practice to improve pacing."))

        # 10. Heatmap: section vs chapter
        if "section" in questions_df.columns:
            heatmap_df1 = questions_df.pivot_table(index="section", columns="chapter", values="isCorrect", aggfunc="count", fill_value=0)
            plt.figure(figsize=(10, 6))
            sns.heatmap(heatmap_df1, annot=True, fmt="d", cmap="YlOrRd")
            plt.title("Section vs Chapter (Question Count)")
            plt.ylabel("Section")
            plt.xlabel("Chapter")
            plt_path = os.path.join(save_dir, "section_vs_chapter_heatmap.png")
            plt.savefig(plt_path, dpi=300, bbox_inches="tight")
            plt.close()
            charts.append((plt_path, "This grid shows how many questions each section had from each chapter. Darker boxes mean more questions, guiding your study focus."))

        # 11. Heatmap: chapter vs level
        heatmap_df2 = questions_df.pivot_table(index="chapter", columns="level", values="isCorrect", aggfunc="count", fill_value=0)
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_df2, annot=True, fmt="d", cmap="YlOrRd")
        plt.title("Chapter vs Level (Question Count)")
        plt.ylabel("Chapter")
        plt.xlabel("Level")
        plt_path = os.path.join(save_dir, "chapter_vs_level_heatmap.png")
        plt.savefig(plt_path, dpi=300, bbox_inches="tight")
        plt.close()
        charts.append((plt_path, "This grid shows how many easy, medium, or hard questions each chapter had. Darker boxes highlight chapters with tough questions to practice."))

        # 12. Heatmap: level vs status
        heatmap_df3 = questions_df.pivot_table(index="level", columns="status", values="isCorrect", aggfunc="count", fill_value=0)
        plt.figure(figsize=(8, 6))
        sns.heatmap(heatmap_df3, annot=True, fmt="d", cmap="YlOrRd")
        plt.title("Level vs Status (Question Count)")
        plt.ylabel("Level")
        plt.xlabel("Status")
        plt_path = os.path.join(save_dir, "level_vs_status_heatmap.png")
        plt.savefig(plt_path, dpi=300, bbox_inches="tight")
        plt.close()
        charts.append((plt_path, "This grid shows if easy, medium, or hard questions were right, wrong, or skipped. Darker boxes for wrong answers show where to improve."))

        # 13. Heatmap: status vs isCorrect
        heatmap_df4 = questions_df.pivot_table(index="status", columns="isCorrect", values="timeTaken", aggfunc="count", fill_value=0)
        plt.figure(figsize=(6, 4))
        sns.heatmap(heatmap_df4, annot=True, fmt="d", cmap="YlOrRd")
        plt.title("Status vs Correctness (Question Count)")
        plt.ylabel("Status")
        plt.xlabel("Correct")
        plt_path = os.path.join(save_dir, "status_vs_correctness_heatmap.png")
        plt.savefig(plt_path, dpi=300, bbox_inches="tight")
        plt.close()
        charts.append((plt_path, "This grid shows if answered questions were correct or incorrect. Darker boxes for incorrect answers highlight areas to review."))

        # 14. Violinplot: section vs timeTaken
        if "section" in questions_df.columns:
            plt.figure(figsize=(8, 5))
            sns.violinplot(data=questions_df, y="section", x="timeTaken", scale="width")
            plt.title("Time Taken Distribution by Section")
            plt.xlabel("Time Taken (s)")
            plt.ylabel("Section")
            plt_path = os.path.join(save_dir, "section_vs_timeTaken_violin.png")
            plt.savefig(plt_path, dpi=300, bbox_inches="tight")
            plt.close()
            charts.append((plt_path, "This chart shows time spent on questions in each section, with wider shapes for varied times. It helps you see where your pacing was uneven."))

        # 15. Violinplot: chapter vs timeTaken
        plt.figure(figsize=(8, 10))
        sns.violinplot(data=questions_df, y="chapter", x="timeTaken", scale="width")
        plt.title("Time Taken Distribution by Chapter")
        plt.xlabel("Time Taken (s)")
        plt.ylabel("Chapter")
        plt_path = os.path.join(save_dir, "chapter_vs_timeTaken_violin.png")
        plt.savefig(plt_path, dpi=300, bbox_inches="tight")
        plt.close()
        charts.append((plt_path, "This chart shows time spent on questions from each chapter, with wider shapes for varied times. It highlights chapters where you were slower."))

        # 16. Violinplot: level vs timeTaken
        plt.figure(figsize=(8, 5))
        sns.violinplot(data=questions_df, y="level", x="timeTaken", scale="width")
        plt.title("Time Taken Distribution by Level")
        plt.xlabel("Time Taken (s)")
        plt.ylabel("Level")
        plt_path = os.path.join(save_dir, "level_vs_timeTaken_violin.png")
        plt.savefig(plt_path, dpi=300, bbox_inches="tight")
        plt.close()
        charts.append((plt_path, "This chart shows time spent on easy, medium, and hard questions, with wider shapes for varied times. It shows which difficulty levels slowed you down."))

        # 17. Violinplot: status vs timeTaken
        plt.figure(figsize=(8, 5))
        sns.violinplot(data=questions_df, y="status", x="timeTaken", scale="width")
        plt.title("Time Taken Distribution by Status")
        plt.xlabel("Time Taken (s)")
        plt.ylabel("Status")
        plt_path = os.path.join(save_dir, "status_vs_timeTaken_violin.png")
        plt.savefig(plt_path, dpi=300, bbox_inches="tight")
        plt.close()
        charts.append((plt_path, "This chart shows time spent on correct, incorrect, or skipped questions, with wider shapes for varied times. It highlights if wrong answers took too long."))

    except Exception as e:
        print(f"Chart generation failed: {e}")
    return charts

# Gemini: Extract chapters
def get_gemini_chapters(json_data):
    syllabus = json_data.get("test", {}).get("syllabus", "")
    sections = json_data.get("sections", [])
    prompt = f"""
You are an expert in processing large JSON files for educational performance analysis. I have a JSON file containing student performance data for a test. The JSON includes:

- A "test.syllabus" field with HTML content listing chapters: {syllabus[:1000]}...
- A "sections" array with questions, where each question has a "chapters" field (e.g., [{{"title": "Functions"}}]).

Extract all unique chapter titles from the JSON, regardless of subject, by:

1. Parsing the "test.syllabus" HTML to identify chapter titles.
2. Supplementing with chapters from "sections.questions.chapters".
3. Ensuring no duplicates and sorting alphabetically.

Output in JSON format as a list:

```json
["Chapter 1", "Chapter 2", ...]
```
"""
    try:
        response = model.generate_content(prompt)
        chapters = json.loads(response.text.strip("```json\n").strip("\n```"))
        return sorted(list(set(chapters)))
    except Exception as e:
        print(f"Gemini chapter extraction failed: {e}")
        # Fallback: Parse syllabus and sections manually
        chapters = set()
        if syllabus:
            soup = BeautifulSoup(syllabus, 'html.parser')
            for li in soup.find_all('li'):
                chapter = li.text.strip()
                if chapter:
                    chapters.add(chapter)
        for section in sections:
            for question in section.get("questions", []):
                for chapter in question.get("questionId", {}).get("chapters", []):
                    chapter_title = chapter.get("title", "").strip()
                    if chapter_title:
                        chapters.add(chapter_title)
        return sorted(list(chapters)) or ["Unknown"]

# Sanitize text for PDF
def sanitize_text(text):
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

# Generate feedback with Gemini
def generate_feedback(questions_df, subject_data, student_name="Student"):
    total = len(questions_df)
    correct = questions_df["isCorrect"].sum()
    accuracy = (correct / total) * 100 if total > 0 else 0
    avg_time = questions_df["timeTaken"].mean()
    time_used = questions_df["timeTaken"].sum() / test_duration * 100

    subject_summary = "\n".join([
        f"- {s['Subject']}: {s.get('TotalCorrect', 0)}/{s.get('TotalAttempted', 0)} correct ({s.get('Accuracy', 0.0):.2f}%), {s.get('TotalTimeTaken', 0):.1f}s"
        for _, s in subject_data.iterrows()
    ])

    chapter_accuracy = questions_df.groupby("chapter")["isCorrect"].mean() * 100
    weakest_chapter = chapter_accuracy.idxmin() if chapter_accuracy.size else "N/A"
    weakest_ch_acc = chapter_accuracy.min() if chapter_accuracy.size else 0
    strongest_chapter = chapter_accuracy.idxmax() if chapter_accuracy.size else "N/A"
    strongest_ch_acc = chapter_accuracy.max() if chapter_accuracy.size else 0

    difficulty_accuracy = questions_df.groupby("level")["isCorrect"].mean() * 100
    toughest_level = difficulty_accuracy.idxmin() if difficulty_accuracy.size else "N/A"
    toughest_level_acc = difficulty_accuracy.min() if difficulty_accuracy.size else 0

    slow_questions = questions_df[questions_df["timeTaken"] > questions_df["timeTaken"].quantile(0.75)]
    slow_acc = slow_questions["isCorrect"].mean() * 100 if not slow_questions.empty else 0

    prompt = f"""
You are an expert educational assistant creating a detailed, motivating feedback report for {student_name} based on their performance in {test_name} ({test_date}). Use the provided data to craft a personalized, encouraging narrative with actionable suggestions generated dynamically based on the performance metrics. All suggestions must be generated by analyzing the data, with no hardcoded or generic advice.

**Performance Data**:
- Total Questions: {total}
- Correct Answers: {correct}
- Total Marks Scored: {correct * (total_marks / total_questions):.2f}/{total_marks}
- Accuracy: {accuracy:.2f}%
- Average Time per Question: {avg_time:.2f}s
- Time Used: {time_used:.2f}% of {test_duration}s
- Weakest Chapter: {weakest_chapter} ({weakest_ch_acc:.2f}%)
- Strongest Chapter: {strongest_chapter} ({strongest_ch_acc:.2f}%)
- Toughest Difficulty: {toughest_level} ({toughest_level_acc:.2f}%)
- Accuracy on Slow Questions: {slow_acc:.2f}%
- Subject-wise Performance:
{subject_summary}

**Instructions**:
- **Intro (100–150 words)**: Greet {student_name}, acknowledge their effort, highlight strengths (e.g., strongest subject or chapter), and encourage improvement in weaker areas.
- **Performance Breakdown (200–300 words)**:
  - **Subject-wise**: Analyze strongest and weakest subjects, including accuracy and time spent, and suggest general strategies for improvement.
  - **Chapter-wise**: Discuss performance by chapter, focusing on weakest and strongest chapters, and provide insights into patterns.
  - **Difficulty-wise**: Evaluate accuracy and time spent across difficulty levels (easy, medium, hard).
  - **Time vs. Accuracy**: Analyze the relationship between time spent and accuracy, highlighting any trends (e.g., slower questions with lower accuracy).
  - **Overall Metrics**: Summarize total marks, time utilization, and accuracy trends.
- **Actionable Suggestions (150–200 words)**: Generate 4–5 specific, actionable suggestions per subject (Physics, Chemistry, Mathematics) and 1 for the Solutions chapter (if applicable). Each suggestion must:
  - Be a separate bullet point starting with '-'.
  - Be tailored to the student's performance data, referencing specific metrics (e.g., accuracy, time spent), chapters, or difficulty levels.
  - Focus on practical, data-driven steps to address weaknesses (e.g., low accuracy in a specific chapter) and reinforce strengths.
  - Avoid generic advice; suggestions must be derived from the provided metrics.
  - Example (do not use directly): If accuracy in Physics is low in Mechanics (40%), suggest targeted practice with Mechanics problems.
- **Tone**: Friendly, encouraging, specific, and motivating.
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
**Solutions Chapter:**
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
        print(f"Gemini API Error: {e}")
        # Fallback: Attempt a second Gemini call for suggestions only
        fallback_prompt = f"""
Generate actionable suggestions for {student_name} based on their performance in {test_name} ({test_date}). Use the provided data to create 4–5 specific, data-driven suggestions per subject (Physics, Chemistry, Mathematics) and 1 for the Solutions chapter (if applicable). Avoid generic advice.

**Performance Data**:
- Total Questions: {total}
- Correct Answers: {correct}
- Accuracy: {accuracy:.2f}%
- Average Time per Question: {avg_time:.2f}s
- Time Used: {time_used:.2f}% of {test_duration}s
- Weakest Chapter: {weakest_chapter} ({weakest_ch_acc:.2f}%)
- Strongest Chapter: {strongest_chapter} ({strongest_ch_acc:.2f}%)
- Toughest Difficulty: {toughest_level} ({toughest_level_acc:.2f}%)
- Accuracy on Slow Questions: {slow_acc:.2f}%
- Subject-wise Performance:
{subject_summary}

**Instructions**:
- Provide 4–5 suggestions per subject (Physics, Chemistry, Mathematics) and 1 for Solutions chapter.
- Each suggestion must:
  - Start with '-'.
  - Reference specific metrics (e.g., accuracy, time, chapters).
  - Be practical and tailored to the data.
- Output only the suggestions in markdown format:
```markdown
**Physics:**
- ...
**Chemistry:**
- ...
**Mathematics:**
- ...
**Solutions Chapter:**
- ...
```
"""
        try:
            fallback_response = model.generate_content(fallback_prompt)
            suggestions_text = fallback_response.text.strip()
            fallback = f"""
### Intro
Dear {student_name}, great effort on {test_name}! Your performance shows potential, especially in {strongest_chapter} ({strongest_ch_acc:.2f}%). Areas like {weakest_chapter} ({weakest_ch_acc:.2f}%) offer growth opportunities. Let's explore your results!

### Performance Breakdown
#### Subject-wise Analysis
{subject_summary}
- Strongest: {subject_data.loc[subject_data['Accuracy'].idxmax(), 'Subject']} ({subject_data['Accuracy'].max():.2f}%).
- Weakest: {subject_data.loc[subject_data['Accuracy'].idxmin(), 'Subject']} ({subject_data['Accuracy'].min():.2f}%).

#### Chapter-wise Analysis
- Strongest: {strongest_chapter} ({strongest_ch_acc:.2f}%).
- Weakest: {weakest_chapter} ({weakest_ch_acc:.2f}%).

#### Difficulty-wise Analysis
- Toughest: {toughest_level} ({toughest_level_acc:.2f}%).

#### Time and Accuracy Insights
- Slow Questions: {slow_acc:.2f}% accuracy.
- Time Used: {time_used:.2f}% ({avg_time:.2f}s/question).

#### Overall Metrics
- Marks: {correct * (total_marks / total_questions):.2f}/{total_marks}
- Accuracy: {accuracy:.2f}%

### Actionable Suggestions
{suggestions_text}
"""
            return sanitize_text(fallback)
        except Exception as e2:
            print(f"Fallback Gemini API Error: {e2}")
            # Ultimate fallback: No suggestions
            return sanitize_text(f"""
### Intro
Dear {student_name}, great effort on {test_name}! Your performance shows potential, especially in {strongest_chapter} ({strongest_ch_acc:.2f}%). Areas like {weakest_chapter} ({weakest_ch_acc:.2f}%) offer growth opportunities. Let's explore your results!

### Performance Breakdown
#### Subject-wise Analysis
{subject_summary}
- Strongest: {subject_data.loc[subject_data['Accuracy'].idxmax(), 'Subject']} ({subject_data['Accuracy'].max():.2f}%).
- Weakest: {subject_data.loc[subject_data['Accuracy'].idxmin(), 'Subject']} ({subject_data['Accuracy'].min():.2f}%).

#### Chapter-wise Analysis
- Strongest: {strongest_chapter} ({strongest_ch_acc:.2f}%).
- Weakest: {weakest_chapter} ({weakest_ch_acc:.2f}%).

#### Difficulty-wise Analysis
- Toughest: {toughest_level} ({toughest_level_acc:.2f}%).

#### Time and Accuracy Insights
- Slow Questions: {slow_acc:.2f}% accuracy.
- Time Used: {time_used:.2f}% ({avg_time:.2f}s/question).

#### Overall Metrics
- Marks: {correct * (total_marks / total_questions):.2f}/{total_marks}
- Accuracy: {accuracy:.2f}%

### Actionable Suggestions
[Unable to generate suggestions due to API issues. Please review performance metrics and focus on weak areas like {weakest_chapter}.]
""")

# Custom PDF class
class PDF(FPDF):
    def __init__(self, logo_path):
        super().__init__()
        self.logo_path = logo_path

    def header(self):
        if self.page_no() > 1:
            if os.path.exists(self.logo_path):
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

    def add_image(self, image_path, caption, description):
        try:
            if os.path.exists(image_path):
                self.image(image_path, x=30, w=150, h=80)
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
                self.multi_cell(0, 6, f"[Image Missing: {os.path.basename(image_path)}]")
            self.ln(6)
        except Exception as e:
            print(f"Image error: {e}")
            self.set_font("Helvetica", "", 10)
            self.set_text_color(255, 0, 0)
            self.multi_cell(0, 6, f"[Image Error: {e}]")

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

def generate_analysis_pdf(questions_df, subject_data, feedback_sections, chapters, image_list, logo_path="logo.png", student_name="Student"):
    pdf = PDF(logo_path)
    pdf.set_auto_page_break(auto=True, margin=15)

    # Cover Page
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(33, 102, 172)
    pdf.cell(0, 20, "[Book] Student Performance Report", align="C", ln=1)
    pdf.ln(10)
    if os.path.exists(logo_path):
        pdf.image(logo_path, x=(210 - 50) / 2, y=80, w=50)
    pdf.ln(60)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 10, f"{test_name} - {test_date}", align="C", ln=1)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, f"For: {student_name}", align="C", ln=1)
    pdf.cell(0, 10, "Generated by MathonGo AI", align="C", ln=1)

    # Summary Statistics
    pdf.add_page()
    pdf.section_title("1. Summary Statistics")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(40, 40, 40)
    total = len(questions_df)
    correct = questions_df["isCorrect"].sum()
    accuracy = (correct / total) * 100 if total > 0 else 0
    avg_time = questions_df["timeTaken"].mean()
    marks_scored = correct * (total_marks / total_questions)
    time_used = questions_df["timeTaken"].sum() / test_duration * 100
    stats = [
        f"[Note] Total Questions: {total}",
        f"[Check] Correct Answers: {correct}",
        f"[Target] Accuracy: {accuracy:.2f}%",
        f"[Timer] Average Time per Question: {avg_time:.2f}s",
        f"[Chart] Marks Scored: {marks_scored:.1f}/{total_marks}",
        f"[Timer] Time Used: {time_used:.2f}% of {test_duration}s"
    ]
    for stat in stats:
        pdf.multi_cell(0, 7, stat)
        pdf.ln(4)

    # Personalized Feedback
    pdf.section_title("2. Personalized Feedback")
    pdf.set_font("Helvetica", "", 10)
    intro_text = sanitize_text(feedback_sections["intro"])
    if intro_text.strip():
        pdf.multi_cell(0, 6, intro_text)
    else:
        pdf.multi_cell(0, 6, "[No feedback provided]")
    pdf.ln(6)

    # Performance Breakdown
    pdf.section_title("3. Performance Analysis")
    pdf.set_font("Helvetica", "", 10)
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
            pdf.multi_cell(0, 6, text)
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
            "Time (s)": "TotalTimeTaken"
        },
        title="Subject-wise Performance"
    )

    # Actionable Suggestions
    pdf.plain_section_title("4. Actionable Suggestions")
    pdf.add_suggestions(sanitize_text(feedback_sections["suggestions"]))

    # Chapter-wise Analysis
    pdf.section_title("5. Chapter-wise Analysis")
    chapter_summary = questions_df.groupby("chapter").agg({
        "isCorrect": ["count", "mean"],
        "timeTaken": "mean"
    }).reset_index()
    chapter_summary.columns = ["chapter", "total_questions", "accuracy", "avg_time"]
    chapter_summary["accuracy"] *= 100
    chapter_summary = chapter_summary.sort_values(by="accuracy", ascending=False)

    if not chapter_summary.empty and chapters and chapters != ["Unknown"]:
        pdf.add_table(
            data=chapter_summary,
            headers=["Chapter", "Questions", "Accuracy (%)", "Avg Time (s)"],
            widths=[70, 30, 30, 30],
            headers_map={
                "Chapter": "chapter",
                "Questions": "total_questions",
                "Accuracy (%)": "accuracy",
                "Avg Time (s)": "avg_time"
            },
            title="Performance by Chapter"
        )
    else:
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(255, 0, 0)
        pdf.multi_cell(0, 6, "[No chapters found in the test data]")
        pdf.ln(4)

    # Visual Insights
    pdf.section_title("6. Visual Insights")
    for i, (image_path, description) in enumerate(image_list):
        if i % 2 == 0 and i > 0:
            pdf.add_page()
        caption = os.path.basename(image_path).replace(".png", "").replace("_", " ").title()
        pdf.add_image(image_path, caption, description)

    # Save and download PDF
    output_path = "Student_Performance_Report.pdf"
    pdf.output(output_path)
    print(f"✅ PDF saved to: {output_path}")
    if os.path.exists(output_path):
        files.download(output_path)
    else:
        print(f"❌ PDF not found at: {output_path}")

# Parse feedback
feedback_raw = generate_feedback(questions_df, subject_data, "Student")
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

# Generate charts
image_list = generate_all_charts(questions_df, save_dir="charts/")

# Generate PDF
try:
    chapters = get_gemini_chapters(data)
    generate_analysis_pdf(questions_df, subject_data, feedback_sections, chapters, image_list, logo_path="logo.png", student_name="Student")
except Exception as e:
    print(f"❌ Error generating PDF: {e}")