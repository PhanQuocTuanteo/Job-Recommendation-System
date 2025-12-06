"""
Career Path Advisor Module
Đề xuất career path dựa trên CV và careerPath.csv
"""
import os
import pandas as pd
from . import resparser
from difflib import SequenceMatcher
from urllib.parse import quote_plus

class CareerPathAdvisor:
    """Career Path Advisor để đề xuất career progression"""
    
    @staticmethod
    def create_google_search_link(job_title):
        """
        Tạo link tìm kiếm Google từ Job Title
        Args:
            job_title: Tên công việc
        Returns:
            URL tìm kiếm trên Google
        """
        if not job_title or pd.isna(job_title):
            return 'https://www.google.com/search?q=jobs'
        
        # URL encode job title
        encoded_title = quote_plus(str(job_title).strip())
        return f'https://www.google.com/search?q={encoded_title}'
    
    def __init__(self, career_path_file='Datasets/careerPath.csv'):
        """
        Initialize career path advisor
        
        Args:
            career_path_file: Path to careerPath.csv
        """
        self.career_path_file = career_path_file
        self.career_paths_df = None
        self._load_career_paths()
    
    def _load_career_paths(self):
        """Load career paths từ CSV"""
        try:
            self.career_paths_df = pd.read_csv(self.career_path_file)
            # Parse Career Path thành list
            self.career_paths_df['career_path_list'] = self.career_paths_df['Career Path'].apply(
                lambda x: [role.strip() for role in str(x).split('>')] if pd.notna(x) else []
            )
            # Normalize skills thành list
            self.career_paths_df['skills_list'] = self.career_paths_df['Skills'].apply(
                lambda x: [s.strip() for s in str(x).split(',')] if pd.notna(x) else []
            )
        except Exception as e:
            print(f"Error loading career paths: {e}")
            self.career_paths_df = pd.DataFrame()
    
    def _similarity(self, str1, str2):
        """Calculate similarity between two strings"""
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
    
    def _match_job_title(self, user_job_title, threshold=0.6):
        """
        Match user job title với job titles trong careerPath.csv
        
        Args:
            user_job_title: Job title từ CV
            threshold: Minimum similarity threshold
        
        Returns:
            Matched job title hoặc None
        """
        if not user_job_title or self.career_paths_df.empty:
            return None
        
        user_job_title_lower = user_job_title.lower()
        best_match = None
        best_score = 0
        
        for _, row in self.career_paths_df.iterrows():
            job_title = str(row['Job Title']).lower()
            
            # Exact match
            if user_job_title_lower == job_title:
                return row['Job Title']
            
            # Contains match
            if user_job_title_lower in job_title or job_title in user_job_title_lower:
                score = self._similarity(user_job_title, row['Job Title'])
                if score > best_score:
                    best_score = score
                    best_match = row['Job Title']
            
            # Similarity match
            score = self._similarity(user_job_title, row['Job Title'])
            if score > best_score and score >= threshold:
                best_score = score
                best_match = row['Job Title']
        
        return best_match if best_score >= threshold else None
    
    def _match_by_skills(self, user_skills):
        """
        Match career path dựa trên skills nếu không tìm thấy job title
        
        Args:
            user_skills: List of user skills
        
        Returns:
            Best matched job title hoặc None
        """
        if not user_skills or self.career_paths_df.empty:
            return None
        
        user_skills_set = set(s.lower().strip() for s in user_skills)
        best_match = None
        best_overlap = 0
        
        for _, row in self.career_paths_df.iterrows():
            path_skills = [s.lower().strip() for s in row['skills_list']]
            path_skills_set = set(path_skills)
            
            # Calculate overlap
            overlap = len(user_skills_set.intersection(path_skills_set))
            overlap_ratio = overlap / len(path_skills_set) if path_skills_set else 0
            
            if overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                best_match = row['Job Title']
        
        # Return match nếu overlap >= 30%
        return best_match if best_overlap >= 0.3 else None
    
    def _extract_job_title_from_resume(self, resume_file):
        """
        Extract job title từ resume (simple approach)
        Tìm trong text, có thể cải thiện sau với NER
        """
        try:
            # Parse resume
            resume_lines = []
            if resume_file.endswith('.pdf'):
                resume_lines = resparser.convert_pdf_to_txt(resume_file)
            elif resume_file.endswith('.docx') or resume_file.endswith('.doc'):
                resume_lines = resparser.convert_docx_to_txt(resume_file)
            
            # Tìm job title patterns trong first few lines
            # Common patterns: "Position:", "Role:", "Title:", hoặc ở đầu experience section
            full_text = ' '.join(resume_lines[:50]).lower()  # First 50 lines
            
            # Tìm các job titles phổ biến trong text
            for _, row in self.career_paths_df.iterrows():
                job_title = str(row['Job Title']).lower()
                # Check nếu job title xuất hiện trong resume
                if job_title in full_text:
                    return row['Job Title']
            
            return None
        except Exception as e:
            print(f"Error extracting job title: {e}")
            return None
    
    def get_career_path(self, resume_file):
        """
        Get career path suggestions cho user
        
        Args:
            resume_file: Path to resume file
        
        Returns:
            Dict với career path information:
            {
                'matched_job_title': str,
                'industry': str,
                'career_path': list of roles,
                'success': bool
            }
        """
        if self.career_paths_df.empty:
            return {
                'success': False,
                'matched_job_title': None,
                'industry': None,
                'career_path': []
            }
        
        try:
            # 1. Extract user skills
            user_skills_data = resparser.skill(resume_file)
            if isinstance(user_skills_data, dict):
                user_skills = user_skills_data.get('skills', [])
            elif isinstance(user_skills_data, list):
                # Handle list format - first element is string of skills
                if user_skills_data and len(user_skills_data) > 0:
                    skills_str = user_skills_data[0]
                    if isinstance(skills_str, str):
                        # Split by space or comma
                        user_skills = [s.strip() for s in skills_str.replace(',', ' ').split() if s.strip()]
                    else:
                        user_skills = []
                else:
                    user_skills = []
            else:
                user_skills = []
            
            # 2. Try extract job title từ resume
            user_job_title = self._extract_job_title_from_resume(resume_file)
            
            # 3. Match với career path
            matched_job_title = None
            matched_row = None
            
            if user_job_title:
                matched_job_title = self._match_job_title(user_job_title)
            
            # 4. Nếu không match bằng job title, thử match bằng skills
            if not matched_job_title:
                matched_job_title = self._match_by_skills(user_skills)
            
            # 5. Get career path
            if matched_job_title:
                matched_row = self.career_paths_df[
                    self.career_paths_df['Job Title'] == matched_job_title
                ].iloc[0]
                
                # Tạo career path với links cho mỗi vị trí
                career_path_with_links = []
                for role in matched_row['career_path_list']:
                    career_path_with_links.append({
                        'title': role,
                        'link': self.create_google_search_link(role)
                    })
                
                return {
                    'success': True,
                    'matched_job_title': matched_job_title,
                    'industry': matched_row['Industry'],
                    'career_path': matched_row['career_path_list'],  # Giữ nguyên để backward compatibility
                    'career_path_with_links': career_path_with_links,  # Mới: có links
                    'skills': matched_row['skills_list']
                }
            else:
                # Return first career path as default (hoặc có thể return None)
                return {
                    'success': False,
                    'matched_job_title': None,
                    'industry': None,
                    'career_path': []
                }
        
        except Exception as e:
            print(f"Error getting career path: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'matched_job_title': None,
                'industry': None,
                'career_path': []
            }

