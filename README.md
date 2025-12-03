# Resume and Job Recommendation System

An intelligent job recommendation system that matches job seekers with opportunities using skills extraction (NLP), experience analysis, and collaborative filtering. Includes career growth suggestions and skill gap identification with course recommendations.

## Features

### Core Features
- **Resume Parsing**: Extract skills, experience, and qualifications from PDF/DOCX resumes using Apache Tika
- **Job Recommendations**: 
  - Content-based filtering using NLP and cosine similarity
  - Collaborative filtering based on user interactions
  - Hybrid recommendation combining both approaches
- **Course Recommendations**: 
  - Recommend courses for missing skills (priority)
  - Recommend advanced courses for existing skills
  - Support for Coursera, Udemy, edX, and Skillshare
- **Career Path Suggestions**: Visualize career progression based on current role and skills
- **User Interaction Tracking**: Track job views and clicks for improved recommendations

### Technical Features
- **NLP Processing**: 
  - spaCy for tokenization, POS tagging, and Named Entity Recognition
  - NLTK for stopwords removal and text preprocessing
  - Skill extraction using Phrase Matching
- **Machine Learning**:
  - TF-IDF vectorization for text similarity
  - Cosine similarity for job matching
  - User-user similarity for collaborative filtering
- **Web Scraping**: Scrape job data from Indeed

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jainiljakasaniya/resume-job-recommendation.git
   cd resume-job-recommendation
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download spaCy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. Start the Flask server:
   ```bash
   python app.py
   ```

5. Access the system in your browser at `http://localhost:5000`

## Usage

### For Job Seekers (Employee)

1. Navigate to the **Employee** page
2. Upload your resume/CV (PDF or DOCX format)
3. View job recommendations:
   - Top 20 job matches based on your skills
   - Skills match percentage
   - Direct links to apply
4. View course recommendations:
   - Courses for missing skills (highlighted in red)
   - Advanced courses for existing skills (highlighted in green)
5. View career path suggestions:
   - Visual progression path based on your current role

### How It Works

1. **Resume Analysis**: 
   - Extract text from resume using Apache Tika
   - Identify skills using NLP techniques
   - Extract work experience and qualifications

2. **Job Matching**:
   - Compare resume skills with job descriptions
   - Calculate similarity using TF-IDF and cosine similarity
   - Consider user interactions (views/clicks) for collaborative filtering
   - Rank jobs by hybrid score (70% content-based + 30% collaborative)

3. **Course Recommendations**:
   - Analyze job requirements to identify missing skills
   - Match courses from multiple platforms
   - Rank by relevance and match score

4. **Career Path**:
   - Match current role from resume
   - Suggest career progression path
   - Display sequential role progression

## Project Structure

```
resume-job-recommendation/
├── app.py                          # Flask application
├── requirements.txt                # Python dependencies
├── templates/
│   ├── index.html                  # Home page
│   └── employee.html               # Employee page
├── static/
│   ├── styles/                     # CSS files
│   ├── img/                        # Images
│   └── java/                       # JavaScript files
├── resume_screening/
│   ├── job.py                      # Job recommendation logic
│   ├── resparser.py                # Resume parsing
│   ├── extract_skill.py            # Skill extraction
│   ├── match.py                    # Text matching utilities
│   ├── course_recommender.py       # Course recommendation
│   ├── collaborative_filter.py     # Collaborative filtering
│   ├── user_interactions.py        # User interaction tracking
│   └── career_path_advisor.py      # Career path suggestions
└── course dataset/                  # Course datasets (CSV files)
```

## Documentation

- **Collaborative Filtering**: See `COLLABORATIVE_FILTERING.md` for details on the collaborative filtering implementation
- **Course Recommendations**: See `COURSE_FEATURE_SUMMARY.md` for course recommendation details

## Technologies Used

- **Backend**: Flask (Python)
- **NLP**: spaCy, NLTK
- **Machine Learning**: scikit-learn (TF-IDF, Cosine Similarity)
- **Data Processing**: Pandas, NumPy
- **File Parsing**: Apache Tika
- **Frontend**: HTML, CSS, JavaScript

## License

Copyright &copy; 2022 - Resume Parser. All Rights Reserved
