"""
Experience Analyzer Module
Extract and analyze work experience from resumes
"""
import re
import os
import pandas as pd
from datetime import datetime
try:
    from dateutil import parser as date_parser
except ImportError:
    # Fallback if dateutil not available
    date_parser = None
from . import resparser
import spacy

# Load spaCy model (reuse existing model if available)
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    # Fallback if model not loaded
    nlp = None

class ExperienceAnalyzer:
    """Analyze work experience from resumes"""
    
    # Keywords for experience section
    EXPERIENCE_KEYWORDS = [
        'experience', 'work experience', 'employment', 'employment history',
        'professional experience', 'work history', 'career', 'positions',
        'employment record', 'work background'
    ]
    
    # Date patterns
    DATE_PATTERNS = [
        r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}',  # Jan 2020
        r'\b\d{1,2}[/-]\d{4}',  # 01/2020 or 1-2020
        r'\b\d{4}[/-]\d{1,2}',  # 2020/01
        r'\b\d{4}',  # 2020
    ]
    
    # Job level keywords
    JUNIOR_KEYWORDS = ['junior', 'entry', 'entry-level', 'intern', 'internship', 'trainee', 'assistant', 'associate']
    SENIOR_KEYWORDS = ['senior', 'lead', 'principal', 'architect', 'manager', 'director', 'head', 'chief', 'vp', 'vice president']
    
    def __init__(self):
        """Initialize Experience Analyzer"""
        self.experiences = []
        self.total_years = 0
        self.level = None
        self.industries = []
    
    def extract_experience(self, resume_file):
        """
        Extract work experience from resume
        
        Args:
            resume_file: Path to resume file
            
        Returns:
            dict: {
                'experiences': list of experience dicts,
                'total_years': float,
                'level': str ('junior', 'mid', 'senior'),
                'industries': list,
                'current_job_title': str
            }
        """
        try:
            # Parse resume text
            resume_lines = []
            if resume_file.endswith('.pdf'):
                resume_lines = resparser.convert_pdf_to_txt(resume_file)
            elif resume_file.endswith('.docx') or resume_file.endswith('.doc'):
                resume_lines = resparser.convert_docx_to_txt(resume_file)
            else:
                return self._empty_result()
            
            if not resume_lines:
                return self._empty_result()
            
            # Find experience section
            experience_section = self._find_experience_section(resume_lines)
            
            # Extract experiences
            experiences = self._parse_experiences(experience_section)
            
            # Calculate metrics
            total_years = self._calculate_years(experiences)
            level = self._determine_level(total_years, experiences)
            industries = self._extract_industries(experiences)
            current_job_title = self._get_current_job_title(experiences)
            
            return {
                'experiences': experiences,
                'total_years': total_years,
                'level': level,
                'industries': industries,
                'current_job_title': current_job_title,
                'success': True
            }
        except Exception as e:
            print(f"Error extracting experience: {e}")
            return self._empty_result()
    
    def _empty_result(self):
        """Return empty result structure"""
        return {
            'experiences': [],
            'total_years': 0.0,
            'level': 'mid',  # Default to mid if can't determine
            'industries': [],
            'current_job_title': None,
            'success': False
        }
    
    def _find_experience_section(self, resume_lines):
        """Find experience section in resume"""
        experience_start = -1
        
        # Find start of experience section
        for i, line in enumerate(resume_lines):
            line_lower = line.lower()
            for keyword in self.EXPERIENCE_KEYWORDS:
                if keyword in line_lower and len(line.strip()) < 50:  # Likely a header
                    experience_start = i
                    break
            if experience_start >= 0:
                break
        
        # If no explicit section found, use first 50% of resume (usually contains experience)
        if experience_start < 0:
            experience_start = len(resume_lines) // 2
        
        # Return experience section (from start to end or next major section)
        return resume_lines[experience_start:]
    
    def _parse_experiences(self, experience_section):
        """Parse individual work experiences"""
        experiences = []
        current_exp = {}
        
        for line in experience_section:
            line = line.strip()
            if not line:
                if current_exp:
                    experiences.append(current_exp)
                    current_exp = {}
                continue
            
            # Try to extract dates
            dates = self._extract_dates(line)
            if dates:
                if 'start_date' not in current_exp:
                    current_exp['start_date'] = dates[0]
                if len(dates) > 1:
                    current_exp['end_date'] = dates[1]
                elif 'end_date' not in current_exp:
                    # Check if it's "Present" or "Current"
                    if re.search(r'\b(present|current|now|till now|to date)\b', line, re.IGNORECASE):
                        current_exp['end_date'] = 'present'
            
            # Try to extract job title (usually first line of experience entry)
            if not current_exp.get('job_title') and len(line) < 100:
                # Check if line looks like a job title (not too long, has capital letters)
                if line[0].isupper() and len(line.split()) < 10:
                    current_exp['job_title'] = line
            
            # Try to extract company name (usually after job title)
            if current_exp.get('job_title') and not current_exp.get('company'):
                # Company name patterns
                if re.search(r'\b(inc|llc|ltd|corp|corporation|company|co\.)\b', line, re.IGNORECASE):
                    current_exp['company'] = line
        
        # Add last experience
        if current_exp:
            experiences.append(current_exp)
        
        # Filter valid experiences (must have job title or dates)
        valid_experiences = []
        for exp in experiences:
            if exp.get('job_title') or exp.get('start_date'):
                valid_experiences.append(exp)
        
        return valid_experiences[:10]  # Limit to 10 most recent
    
    def _extract_dates(self, text):
        """Extract dates from text"""
        dates = []
        
        for pattern in self.DATE_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    # Try to parse date
                    if len(match) == 4:  # Just year
                        date_obj = datetime(int(match), 1, 1)
                    else:
                        if date_parser:
                            date_obj = date_parser.parse(match, fuzzy=True)
                        else:
                            # Fallback: extract year
                            year_match = re.search(r'\d{4}', match)
                            if year_match:
                                date_obj = datetime(int(year_match.group()), 1, 1)
                            else:
                                continue
                    dates.append(date_obj)
                except:
                    continue
        
        return sorted(dates) if dates else []
    
    def _calculate_years(self, experiences):
        """Calculate total years of experience"""
        if not experiences:
            return 0.0
        
        total_months = 0
        
        for exp in experiences:
            start_date = exp.get('start_date')
            end_date = exp.get('end_date')
            
            if not start_date:
                continue
            
            if end_date == 'present' or end_date is None:
                end_date = datetime.now()
            elif isinstance(end_date, str):
                if date_parser:
                    try:
                        end_date = date_parser.parse(end_date, fuzzy=True)
                    except:
                        end_date = datetime.now()
                else:
                    # Fallback: try to parse year only
                    year_match = re.search(r'\d{4}', end_date)
                    if year_match:
                        try:
                            end_date = datetime(int(year_match.group()), 1, 1)
                        except:
                            end_date = datetime.now()
                    else:
                        end_date = datetime.now()
            
            if isinstance(start_date, str):
                if date_parser:
                    try:
                        start_date = date_parser.parse(start_date, fuzzy=True)
                    except:
                        continue
                else:
                    # Fallback: try to parse year only
                    year_match = re.search(r'\d{4}', start_date)
                    if year_match:
                        try:
                            start_date = datetime(int(year_match.group()), 1, 1)
                        except:
                            continue
                    else:
                        continue
            
            if isinstance(start_date, datetime) and isinstance(end_date, datetime):
                delta = end_date - start_date
                total_months += delta.days / 30.0
        
        return round(total_months / 12.0, 1)
    
    def _determine_level(self, years_experience, experiences):
        """Determine experience level"""
        # Check job titles for level indicators
        job_titles_text = ' '.join([exp.get('job_title', '') for exp in experiences]).lower()
        
        # Check for senior keywords
        for keyword in self.SENIOR_KEYWORDS:
            if keyword in job_titles_text:
                return 'senior'
        
        # Check for junior keywords
        for keyword in self.JUNIOR_KEYWORDS:
            if keyword in job_titles_text:
                return 'junior'
        
        # Determine by years of experience
        if years_experience < 2:
            return 'junior'
        elif years_experience >= 5:
            return 'senior'
        else:
            return 'mid'
    
    def _extract_industries(self, experiences):
        """Extract industries from experiences"""
        industries = []
        
        # Common industry keywords
        industry_keywords = {
            'technology': ['tech', 'software', 'it', 'computer', 'developer', 'programming'],
            'finance': ['finance', 'banking', 'investment', 'financial', 'accounting'],
            'healthcare': ['health', 'medical', 'hospital', 'clinic', 'pharmaceutical'],
            'education': ['education', 'school', 'university', 'teaching', 'academic'],
            'retail': ['retail', 'sales', 'store', 'shop'],
            'consulting': ['consulting', 'consultant', 'advisory'],
            'manufacturing': ['manufacturing', 'production', 'factory'],
        }
        
        all_text = ' '.join([
            exp.get('job_title', '') + ' ' + exp.get('company', '')
            for exp in experiences
        ]).lower()
        
        for industry, keywords in industry_keywords.items():
            for keyword in keywords:
                if keyword in all_text:
                    if industry not in industries:
                        industries.append(industry)
                    break
        
        return industries
    
    def _get_current_job_title(self, experiences):
        """Get current/most recent job title"""
        if not experiences:
            return None
        
        # Get first experience (usually most recent)
        first_exp = experiences[0]
        return first_exp.get('job_title')
    
    def filter_jobs_by_experience(self, job_recommendations_df, experience_data):
        """
        Filter and adjust job rankings based on experience level
        
        Args:
            job_recommendations_df: DataFrame with job recommendations
            experience_data: dict from extract_experience()
            
        Returns:
            DataFrame with adjusted rankings
        """
        if job_recommendations_df.empty or not experience_data.get('success'):
            return job_recommendations_df
        
        level = experience_data.get('level', 'mid')
        years = experience_data.get('total_years', 0)
        
        # Add experience match score
        job_recommendations_df = job_recommendations_df.copy()
        job_recommendations_df['Experience Match'] = 0.0
        
        # Analyze job titles for level indicators
        for idx, row in job_recommendations_df.iterrows():
            job_title = str(row.get('Job Title', '')).lower()
            match_score = 1.0  # Default match
            
            # Check level mismatch
            if level == 'junior':
                # Junior candidates: prefer entry-level jobs, penalize senior jobs
                if any(kw in job_title for kw in self.SENIOR_KEYWORDS):
                    match_score = 0.7  # Reduce score for senior positions
            elif level == 'senior':
                # Senior candidates: prefer senior jobs, penalize junior jobs
                if any(kw in job_title for kw in self.JUNIOR_KEYWORDS):
                    match_score = 0.6  # Reduce score for junior positions
            
            # Adjust based on years of experience
            if years < 1:
                # Very new: prefer entry-level
                if any(kw in job_title for kw in self.JUNIOR_KEYWORDS):
                    match_score = 1.0
                else:
                    match_score *= 0.8
            elif years > 8:
                # Very experienced: prefer senior roles
                if any(kw in job_title for kw in self.SENIOR_KEYWORDS):
                    match_score = 1.0
                elif any(kw in job_title for kw in self.JUNIOR_KEYWORDS):
                    match_score *= 0.5
            
            job_recommendations_df.at[idx, 'Experience Match'] = match_score
        
        # Combine with existing Skills Match
        if 'Skills Match' in job_recommendations_df.columns:
            job_recommendations_df['Skills Match'] = pd.to_numeric(
                job_recommendations_df['Skills Match'], errors='coerce'
            )
            # Weighted combination: Skills (80%) + Experience (20%)
            job_recommendations_df['Combined Score'] = (
                job_recommendations_df['Skills Match'] * 0.8 +
                job_recommendations_df['Experience Match'] * 100 * 0.2
            )
            # Sort by combined score
            job_recommendations_df = job_recommendations_df.sort_values(
                'Combined Score', ascending=False
            ).reset_index(drop=True)
            # Drop temporary columns
            job_recommendations_df = job_recommendations_df.drop(
                columns=['Experience Match', 'Combined Score']
            )
        
        return job_recommendations_df

