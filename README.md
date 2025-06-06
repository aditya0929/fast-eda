# fast-eda

# Student Performance Analysis Dashboard

Welcome to the **Student Performance Analysis Dashboard**, a powerful web application built with **Streamlit** to analyze student test performance data. This tool transforms raw JSON data into interactive visualizations, detailed insights, and professional PDF reports, helping students and educators identify strengths, weaknesses, and actionable improvement areas.



## 🚀 Features

- 📊 **Interactive Visualizations**: Generate 17 types of charts (histograms, heatmaps, violin plots, etc.) to explore performance patterns across subjects, chapters, difficulty levels, and time usage.  
- 📈 **Subject & Chapter Analysis**: Detailed breakdowns of performance by subject and chapter, with metrics like accuracy and average time per question.  
- ⏱️ **Time Management Insights**: Analyze time spent per question, section, or chapter to optimize pacing.  
- 📄 **PDF Report Generation**: Create professional, customizable PDF reports with summary statistics, personalized feedback, tables, and visualizations.  
- 🎯 **Personalized Feedback**: AI-powered (via Google Gemini) or fallback feedback with actionable, chapter-specific suggestions for improvement.  
- 👤 **User-Friendly Interface**: Streamlit-powered dashboard with a sidebar for file uploads, navigation, and student name input.

---

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

## 📈 Project Journey

### Initial Approach: Colab-Based Analysis

The project began with a `student_performance_analysis.py` script in Google Colab:

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
