# fast-eda

# Student Performance Analysis Dashboard

Welcome to the **Student Performance Analysis Dashboard**, a powerful web application built with **Streamlit** to analyze student test performance data. This tool transforms raw JSON data into interactive visualizations, detailed insights, and professional PDF reports, helping students and educators identify strengths, weaknesses, and actionable improvement areas.
![image](https://github.com/user-attachments/assets/be23767a-002f-44cc-9418-6ef5065c4774)



## 🚀 Features

- 📊 **Interactive Visualizations**: Generate 17 types of charts (histograms, heatmaps, violin plots, etc.) to explore performance patterns across subjects, chapters, difficulty levels, and time usage.  
- 📈 **Subject & Chapter Analysis**: Detailed breakdowns of performance by subject and chapter, with metrics like accuracy and average time per question.  
- ⏱️ **Time Management Insights**: Analyze time spent per question, section, or chapter to optimize pacing.  
- 📄 **PDF Report Generation**: Create professional, customizable PDF reports with summary statistics, personalized feedback, tables, and visualizations.  
- 🎯 **Personalized Feedback**: AI-powered (via Google Gemini) or fallback feedback with actionable, chapter-specific suggestions for improvement.  
- 👤 **User-Friendly Interface**: Streamlit-powered dashboard with a sidebar for file uploads, navigation, and student name input.

---
![image](https://github.com/user-attachments/assets/16b9160a-60db-4d91-9cfd-88f7de0bd394)
![image](https://github.com/user-attachments/assets/c0c01567-a7db-424b-9217-1662499d7717)


## 🧰 Tech Stack

- **Python**: Core programming language  
- **Streamlit**: Web application framework  
- **Pandas & NumPy**: Data manipulation and analysis  
- **Seaborn & Matplotlib**: Data visualization  
- **FPDF**: PDF generation  
- **Google Gemini API**: AI-powered feedback and chapter extraction  
- **BeautifulSoup**: HTML parsing for syllabus data  
- **Bootstrap & Custom CSS**: Dashboard styling  

---
![image](https://github.com/user-attachments/assets/a3021bac-9461-4a4a-8b60-f855ce45e9ea)
![image](https://github.com/user-attachments/assets/8cc09c61-3619-4206-a399-2b8a4118917f)


## chapter extraction prompt 
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

{{
  "Physics": ["Chapter 1", "Chapter 2", ...],
  "Chemistry": ["Chapter 1", "Chapter 2", ...],
  "Mathematics": ["Chapter 1", "Chapter 2", ...]
}}

## feedback extraction prompt 

### Feedback Generation Prompt

This prompt is used to generate personalized feedback using the Gemini API:


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
- 
- **Output Format**:
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


## How Prompts Help

- **Chapter Extraction Prompt**: This prompt helps in accurately extracting and categorizing chapters from the JSON data, ensuring that each chapter is associated with the correct subject. This structured data is crucial for generating subject-wise and chapter-wise performance analysis.

- **Feedback Generation Prompt**: This prompt aids in creating personalized and actionable feedback for students. By leveraging detailed performance data, the prompt ensures that the feedback is specific, motivating, and tailored to the student's strengths and weaknesses. This helps students understand their performance better and provides clear guidance on areas for improvement.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License.


## 📈 Project Journey

### Initial Approach: Colab-Based Analysis

The project began with a `main.py` script in Google Colab:

- **JSON Parsing**: Extracted test details, subject performance, and question-level data  
- **Data Analysis**: Used Pandas for computing metrics like accuracy and time taken  
- **Visualizations**: Generated 17 charts using Seaborn and Matplotlib  
- **PDF Generation**: Included tables, charts, and AI-generated feedback using FPDF and Gemini  
- **Feedback Mechanism**: AI-generated chapter-specific feedback with a fallback strategy  

**Limitations**: Manual file uploads, no interactivity, Colab dependency

> ✅ Prompt 1: *Remove Subject Column (Colab)*  
> Approach: Grouped chapter summary by `chapter` only. Updated headers and widths in the table.  
> Result: PDF report correctly excluded the "Subject" column.

---

### Transition to Streamlit

The project moved to a full-fledged **Streamlit dashboard**:

- Implemented **file uploader** in the sidebar  
- Added **session state** to persist data  
- Created **section navigation** via sidebar radio buttons  
- Converted plots to **in-memory** images (BytesIO)  
- Integrated **PDF download** with `download_button`  
- Applied **custom CSS** for modern, Bootstrap-like design

> ✅ Prompt 2: *Transition to Streamlit*  
> Result: All Colab functionality ported successfully with improved UI/UX.

---

### Prompts and Iterations

> ✅ Prompt 3: *Create README*  
> Goal: Summarize project development, structure, and features for GitHub.

---

## 🧠 Key Design Decisions

- **Modularity**: Functions like `parse_json_data`, `generate_all_charts`, and `generate_feedback` reused across environments  
- **Robustness**: Handled Gemini API failures with fallback logic  
- **User Experience**: Clean UI, intuitive flow, clear error messaging  
- **Extensibility**: Future-ready for new metrics, visualizations, or APIs

---

## 💻 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/student-performance-analysis.git
cd student-performance-analysis
