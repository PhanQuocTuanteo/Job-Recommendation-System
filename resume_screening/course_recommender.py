"""
Course Recommendation Module
Đề xuất khóa học dựa trên missing skills và skills hiện có của user
"""
import os
import pandas as pd
import json
import re
from collections import Counter
from . import match, extract_skill, resparser

# Global cache để tránh load lại datasets mỗi lần
_course_recommender_cache = None

class CourseRecommender:
    """
    Đề xuất khóa học dựa trên CV và job recommendations
    """
    
    def __init__(self, course_data_dir='course dataset', use_cache=True):
        """
        Initialize course recommender
        
        Args:
            course_data_dir: Directory chứa course datasets
            use_cache: Sử dụng cached instance nếu có
        """
        global _course_recommender_cache
        
        self.course_data_dir = course_data_dir
        self.all_courses = None
        self.jobs_df = None
        
        # Use cached instance nếu có
        if use_cache and _course_recommender_cache is not None:
            self.all_courses = _course_recommender_cache.all_courses
            self.jobs_df = _course_recommender_cache.jobs_df
        else:
            self._load_jobs_data()
            self._load_all_courses()
            # Cache instance
            if use_cache:
                _course_recommender_cache = self
    
    def _load_jobs_data(self):
        """Load job data để extract skills từ job descriptions"""
        try:
            self.jobs_df = pd.read_csv('indeed_data.csv')
        except:
            self.jobs_df = pd.DataFrame()
    
    def _load_all_courses(self):
        """Load và merge tất cả course datasets (optimized - chỉ extract skills khi cần)"""
        print("Loading course datasets...")
        courses_list = []
        
        # Load Coursera (có skills sẵn trong JSON)
        try:
            print("Loading Coursera...")
            coursera = pd.read_csv(f'{self.course_data_dir}/Coursera.csv')
            coursera['platform'] = 'Coursera'
            coursera['skills'] = coursera['skills'].apply(self._parse_coursera_skills)
            coursera['title'] = coursera['course']
            # Coursera không có link trong dataset, tạo search link
            coursera['link'] = coursera['course'].apply(lambda x: f'https://www.coursera.org/search?query={x.replace(" ", "%20")}')
            coursera['description'] = coursera['course']
            courses_list.append(coursera)
            print(f"✓ Coursera loaded: {len(coursera)} courses")
        except Exception as e:
            print(f"Warning: Could not load Coursera: {e}")
        
        # Load edX (có skills sẵn và có link)
        try:
            print("Loading edX...")
            edx = pd.read_csv(f'{self.course_data_dir}/edx.csv')
            edx['platform'] = 'edX'
            edx['skills'] = edx['associatedskills'].apply(self._parse_edx_skills)
            edx['title'] = edx['title']
            edx['description'] = edx['title']
            # edX có cột 'link' sẵn, giữ nguyên
            if 'link' not in edx.columns:
                edx['link'] = edx['title'].apply(lambda x: f'https://www.edx.org/search?q={x.replace(" ", "%20")}')
            edx['rating'] = None
            edx['reviewcount'] = None
            courses_list.append(edx)
            print(f"✓ edX loaded: {len(edx)} courses")
        except Exception as e:
            print(f"Warning: Could not load edX: {e}")
        
        # Load Udemy (cần extract skills - chỉ sample để tăng tốc)
        try:
            print("Loading Udemy (sampling for speed)...")
            udemy_full = pd.read_csv(f'{self.course_data_dir}/Udemy.csv')
            # Sample để tăng tốc (lấy top rated courses)
            if 'rating' in udemy_full.columns:
                udemy = udemy_full.nlargest(5000, 'rating')  # Top 5000 courses
            else:
                udemy = udemy_full.head(5000)  # First 5000
            
            udemy['platform'] = 'Udemy'
            # Chỉ extract skills từ title thay vì description (nhanh hơn)
            udemy['skills'] = udemy['title'].apply(self._extract_skills_from_text)
            # Udemy không có link trong dataset, tạo search link
            udemy['link'] = udemy['title'].apply(lambda x: f'https://www.udemy.com/courses/search/?q={x.replace(" ", "%20")}')
            courses_list.append(udemy)
            print(f"✓ Udemy loaded: {len(udemy)} courses (sampled)")
        except Exception as e:
            print(f"Warning: Could not load Udemy: {e}")
        
        # Load Skillshare (sample để tăng tốc)
        try:
            print("Loading Skillshare (sampling for speed)...")
            skillshare_full = pd.read_csv(f'{self.course_data_dir}/skillshare.csv')
            skillshare = skillshare_full.head(2000)  # First 2000 courses
            skillshare['platform'] = 'Skillshare'
            skillshare['skills'] = skillshare['title'].apply(self._extract_skills_from_text)
            skillshare['title'] = skillshare['title']
            skillshare['description'] = skillshare['title']
            # Skillshare có cột 'link' sẵn, giữ nguyên
            if 'link' not in skillshare.columns:
                skillshare['link'] = skillshare['title'].apply(lambda x: f'https://www.skillshare.com/en/search?query={x.replace(" ", "%20")}')
            skillshare['rating'] = None
            skillshare['reviewcount'] = None
            skillshare['level'] = None
            courses_list.append(skillshare)
            print(f"✓ Skillshare loaded: {len(skillshare)} courses (sampled)")
        except Exception as e:
            print(f"Warning: Could not load Skillshare: {e}")
        
        # Merge tất cả courses
        if courses_list:
            print("Merging courses...")
            self.all_courses = pd.concat(courses_list, ignore_index=True)
            # Normalize columns
            self.all_courses = self._normalize_courses(self.all_courses)
            
            # Preprocess course skills thành text (GIỐNG JOB RECOMMENDATIONS)
            # Để match nhanh hơn, không cần process lại mỗi lần
            print("Preprocessing course skills for fast matching...")
            def preprocess_skills_text(skills):
                if not isinstance(skills, list):
                    skills = []
                skills_text = ' '.join(str(s).lower() for s in skills)
                return match.preprocessing(skills_text)
            
            self.all_courses['skills_text'] = self.all_courses['skills'].apply(preprocess_skills_text)
            print(f"✓ Total courses loaded: {len(self.all_courses)}")
        else:
            self.all_courses = pd.DataFrame()
            print("⚠ No courses loaded!")
    
    def _parse_coursera_skills(self, skills_json):
        """Parse JSON skills từ Coursera"""
        if pd.isna(skills_json):
            return []
        try:
            skills = json.loads(skills_json)
            return [s.strip().lower() for s in skills if s.strip()]
        except:
            return []
    
    def _parse_edx_skills(self, skills_str):
        """Parse comma-separated skills từ edX"""
        if pd.isna(skills_str):
            return []
        try:
            skills = [s.strip().lower() for s in str(skills_str).split(',') if s.strip()]
            return skills
        except:
            return []
    
    def _extract_skills_from_text(self, text):
        """Extract skills từ text sử dụng spaCy PhraseMatcher"""
        if pd.isna(text):
            return []
        try:
            text_str = str(text)
            skills = extract_skill.extract_skills(text_str)
            return [s.lower() for s in skills]
        except:
            return []
    
    def _normalize_courses(self, df):
        """Normalize course dataframe về cùng format"""
        # Select và rename columns
        columns_map = {
            'title': 'title',
            'platform': 'platform',
            'skills': 'skills',
            'rating': 'rating',
            'reviewcount': 'reviewcount',
            'level': 'level',
            'duration': 'duration',
            'link': 'link',
            'description': 'description'
        }
        
        # Create normalized dataframe
        normalized = pd.DataFrame()
        for col in ['title', 'platform', 'skills', 'rating', 'reviewcount', 'level', 'duration', 'link', 'description']:
            if col in df.columns:
                normalized[col] = df[col]
            else:
                normalized[col] = None
        
        # Fill missing values
        normalized['skills'] = normalized['skills'].apply(lambda x: x if isinstance(x, list) else [])
        normalized['rating'] = pd.to_numeric(normalized['rating'], errors='coerce')
        normalized['reviewcount'] = normalized['reviewcount'].apply(self._normalize_review_count)
        
        return normalized
    
    def _normalize_review_count(self, count):
        """Normalize review count (handle '16.4k' format)"""
        if pd.isna(count):
            return 0
        try:
            count_str = str(count).lower().replace(',', '')
            if 'k' in count_str:
                return float(count_str.replace('k', '')) * 1000
            return float(count_str)
        except:
            return 0
    
    def extract_missing_skills(self, user_skills, job_recommendations_df):
        """
        Extract missing skills từ job recommendations
        
        Args:
            user_skills: List of user's skills
            job_recommendations_df: DataFrame với job recommendations (có column 'Link')
        
        Returns:
            List of missing skills
        """
        if self.jobs_df.empty or job_recommendations_df.empty:
            return []
        
        # Get top 5 jobs
        top_jobs = job_recommendations_df.head(5)
        
        # Extract skills từ job descriptions
        all_required_skills = []
        for _, job_row in top_jobs.iterrows():
            # Access Series values correctly
            job_link = job_row['Link'] if 'Link' in job_row.index else ''
            if not job_link or pd.isna(job_link):
                continue
            
            # Find job description từ jobs_df
            job_desc_row = self.jobs_df[self.jobs_df['link'] == job_link]
            if not job_desc_row.empty:
                job_desc_series = job_desc_row.iloc[0]
                job_description = job_desc_series['description'] if 'description' in job_desc_series.index else ''
                if job_description and str(job_description) != 'nan':
                    try:
                        # Extract skills từ description
                        required_skills = extract_skill.extract_skills(str(job_description))
                        all_required_skills.extend([s.lower() for s in required_skills if s])
                    except Exception as e:
                        print(f"Error extracting skills from job description: {e}")
                        continue
        
        # Count frequency
        skill_counter = Counter(all_required_skills)
        
        # Find missing skills
        user_skills_set = set(s.lower() for s in user_skills)
        missing_skills = []
        
        for skill, freq in skill_counter.most_common(20):
            if skill.lower() not in user_skills_set and skill:
                missing_skills.append(skill)
        
        return missing_skills[:10]  # Top 10 missing skills
    
    def match_courses_by_skills(self, target_skills, courses_df=None, top_n=15):
        """
        Match courses với target skills - OPTIMIZED: dùng vectorization giống job recommendations
        
        Args:
            target_skills: List of target skills
            courses_df: DataFrame với courses (None = use all_courses)
            top_n: Number of courses to return
        
        Returns:
            DataFrame với matched courses và scores
        """
        if courses_df is None:
            courses_df = self.all_courses.copy()
        
        if courses_df.empty or not target_skills:
            return pd.DataFrame()
        
        try:
            # Convert target skills to text (giống như job recommendations)
            target_skills_text = ' '.join(str(s).lower() for s in target_skills)
            target_skills_text = match.preprocessing(target_skills_text)
            
            # Sử dụng skills_text đã được preprocess sẵn (NHANH HƠN NHIỀU!)
            if 'skills_text' in courses_df.columns:
                course_skills_texts = courses_df['skills_text'].fillna('').tolist()
            else:
                # Fallback: preprocess nếu chưa có
                course_skills_texts = []
                for _, course in courses_df.iterrows():
                    course_skills = course['skills'] if 'skills' in course.index else []
                    if not isinstance(course_skills, list):
                        course_skills = []
                    skills_text = ' '.join(str(s).lower() for s in course_skills)
                    course_skills_texts.append(match.preprocessing(skills_text))
            
            if not course_skills_texts:
                return pd.DataFrame()
            
            # Vectorize và tính cosine similarity (GIỐNG JOB RECOMMENDATIONS - RẤT NHANH!)
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Combine target skills với all course skills
            all_texts = [target_skills_text] + course_skills_texts
            
            # Vectorize (giống như job recommendations)
            cv = CountVectorizer()
            count_matrix = cv.fit_transform(all_texts)
            
            # Calculate cosine similarity (vectorized - nhanh!)
            cosine_sim = cosine_similarity(count_matrix[0:1], count_matrix[1:])
            similarity_scores = cosine_sim[0] * 100  # Convert to percentage
            
            # Add scores directly to dataframe (NHANH HƠN merge)
            courses_df = courses_df.copy()
            courses_df['skill_match'] = [round(score, 2) for score in similarity_scores]
            
            # Calculate additional factors
            rating = courses_df['rating'].fillna(0) if 'rating' in courses_df.columns else pd.Series([0] * len(courses_df))
            review_count = courses_df['reviewcount'].fillna(0) if 'reviewcount' in courses_df.columns else pd.Series([0] * len(courses_df))
            
            # Normalize review count
            max_reviews = review_count.max() if review_count.max() > 0 else 1
            review_score = (review_count / max_reviews * 100).clip(0, 100)
            
            # Combined score
            courses_df['combined_score'] = (
                courses_df['skill_match'] * 0.6 +  # Skill match: 60%
                (rating * 20) * 0.2 +             # Rating: 20%
                review_score * 0.2                 # Reviews: 20%
            ).round(2)
            
            # Filter courses with match score > 0 và sort
            courses_with_scores = courses_df[
                courses_df['skill_match'] > 0
            ].sort_values('combined_score', ascending=False).reset_index(drop=True)
            
            return courses_with_scores.head(top_n)
            
        except Exception as e:
            print(f"Error in vectorized course matching: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def recommend_courses_by_missing_skills(self, user_skills, job_recommendations_df, top_n=15):
        """
        Đề xuất courses dựa trên missing skills từ job recommendations
        
        Args:
            user_skills: List of user's skills
            job_recommendations_df: DataFrame với job recommendations
            top_n: Number of courses to return
        
        Returns:
            DataFrame với recommended courses
        """
        # Extract missing skills
        missing_skills = self.extract_missing_skills(user_skills, job_recommendations_df)
        
        if not missing_skills:
            return pd.DataFrame()
        
        # Match courses
        recommended = self.match_courses_by_skills(missing_skills, top_n=top_n)
        
        return recommended
    
    def recommend_courses_by_existing_skills(self, user_skills, top_n=15):
        """
        Đề xuất courses nâng cao cho skills hiện có
        
        Args:
            user_skills: List of user's skills
            top_n: Number of courses to return
        
        Returns:
            DataFrame với recommended courses
        """
        if not user_skills:
            return pd.DataFrame()
        
        # Match courses với existing skills (advanced courses)
        recommended = self.match_courses_by_skills(user_skills, top_n=top_n * 2)
        
        # Filter để chỉ lấy courses có level Advanced hoặc Intermediate
        if not recommended.empty and 'level' in recommended.columns:
            try:
                recommended = recommended[
                    recommended['level'].astype(str).str.contains('Advanced|Intermediate|All Levels', case=False, na=False)
                ]
            except:
                # If filtering fails, just return all
                pass
        
        return recommended.head(top_n) if not recommended.empty else pd.DataFrame()
    
    def recommend_courses(self, user_skills, job_recommendations_df, top_n=20):
        """
        Main function: Recommend courses dựa trên cả missing skills và existing skills
        
        Args:
            user_skills: List of user's skills
            job_recommendations_df: DataFrame với job recommendations
            top_n: Total number of courses to return
        
        Returns:
            DataFrame với recommended courses
        """
        # Check if courses are loaded
        if self.all_courses is None or self.all_courses.empty:
            print("Warning: No courses loaded")
            return pd.DataFrame()
        
        all_recommended = []
        
        # 1. Courses for missing skills (priority)
        try:
            missing_skills_courses = self.recommend_courses_by_missing_skills(
                user_skills, 
                job_recommendations_df, 
                top_n=top_n // 2
            )
            if not missing_skills_courses.empty:
                missing_skills_courses['recommendation_type'] = 'Missing Skills'
                all_recommended.append(missing_skills_courses)
        except Exception as e:
            print(f"Error in missing skills recommendations: {e}")
        
        # 2. Advanced courses for existing skills
        try:
            existing_skills_courses = self.recommend_courses_by_existing_skills(
                user_skills, 
                top_n=top_n // 2
            )
            if not existing_skills_courses.empty:
                existing_skills_courses['recommendation_type'] = 'Advanced Learning'
                all_recommended.append(existing_skills_courses)
        except Exception as e:
            print(f"Error in existing skills recommendations: {e}")
        
        # Combine và remove duplicates
        if all_recommended:
            try:
                combined = pd.concat(all_recommended, ignore_index=True)
                
                if combined.empty:
                    return pd.DataFrame()
                
                # Remove duplicates based on title
                if 'title' in combined.columns:
                    combined = combined.drop_duplicates(subset='title', keep='first')
                
                # Sort by combined_score
                if 'combined_score' in combined.columns:
                    combined = combined.sort_values('combined_score', ascending=False)
                
                # Format output - check which columns exist
                required_columns = ['title', 'platform', 'skills', 'rating', 'level', 
                                  'duration', 'link', 'skill_match', 'recommendation_type']
                
                # Add missing columns with default values
                for col in required_columns:
                    if col not in combined.columns:
                        combined[col] = None
                
                output = combined[required_columns].copy()
                
                # Format skills column safely
                def format_skills(x):
                    if isinstance(x, list) and x:
                        return ', '.join(str(s) for s in x[:5])
                    elif isinstance(x, str):
                        return x
                    else:
                        return 'N/A'
                
                output['skills'] = output['skills'].apply(format_skills)
                
                # Fill NaN values (nhưng giữ link nếu có)
                # Đảm bảo link không bị fillna nếu đã có giá trị
                for col in output.columns:
                    if col == 'link':
                        # Chỉ fillna link nếu thực sự là NaN hoặc empty string
                        output[col] = output[col].apply(lambda x: x if x and str(x) != 'nan' and str(x).strip() != '' else 'N/A')
                    else:
                        output[col] = output[col].fillna('N/A')
                
                # Rename columns
                output.columns = [
                    'Course Title', 'Platform', 'Skills Covered', 'Rating', 
                    'Level', 'Duration', 'Link', 'Match Score', 'Recommendation Type'
                ]
                
                return output.head(top_n)
            except Exception as e:
                print(f"Error formatting course recommendations: {e}")
                import traceback
                traceback.print_exc()
                return pd.DataFrame()
        
        return pd.DataFrame()

