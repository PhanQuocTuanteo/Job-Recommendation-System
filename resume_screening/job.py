import os
import pandas as pd
from . import resparser, match, course_recommender
import nltk
from nltk.corpus import stopwords
from . import indeed_web_scraping_using_bs4
from . import collaborative_filter, user_interactions
from . import career_path_advisor
from . import experience_analyzer
from urllib.parse import quote_plus

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('maxent_ne_chunker')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('brown')

stopw  = set(stopwords.words('english'))

def create_indeed_search_link(job_title):
    """
    Tạo link tìm kiếm Indeed từ Job Title
    Args:
        job_title: Tên công việc
    Returns:
        URL tìm kiếm trên Indeed
    """
    if not job_title or pd.isna(job_title):
        return 'https://in.indeed.com/jobs'
    
    # URL encode job title (thay khoảng trắng bằng +)
    encoded_title = quote_plus(str(job_title).strip())
    return f'https://in.indeed.com/jobs?q={encoded_title}'

def find_sort_job(f):
    """
    Content-based job recommendation (ORIGINAL FUNCTION - KHÔNG THAY ĐỔI)
    """
    job = pd.read_csv(r'Datasets/indeed_data.csv')
    job['test'] = job['description'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in (stopw)]))
    df = job.drop_duplicates(subset='test').reset_index(drop=True)
    df['clean'] = df['test'].apply(match.preprocessing)
    jobdesc = (df['clean'].values.astype('U'))
    skills = resparser.skill(f)
    # skills = ' '.join(word for word in skills['skills'])
    skills = match.preprocessing(skills[0])
    # del skills[0]
    count_matrix = match.vectorizing(skills, jobdesc)
    matchPercentage = match.coSim(count_matrix)
    matchPercentage = pd.DataFrame(matchPercentage, columns=['Skills Match'])
    #Job Vacancy Recommendations
    result_cosine = df[['title','company','link']]
    result_cosine = result_cosine.join(matchPercentage)
    result_cosine = result_cosine[['title','company','Skills Match','link']]
    result_cosine.columns = ['Job Title','Company','Skills Match','Link']
    # Lưu link gốc để dùng cho collaborative filtering
    result_cosine['Original Link'] = result_cosine['Link'].copy()
    # Tạo link tìm kiếm Indeed từ Job Title thay vì dùng link gốc (có thể đã hết hạn)
    result_cosine['Link'] = result_cosine['Job Title'].apply(create_indeed_search_link)
    
    result_cosine = result_cosine.sort_values('Skills Match', ascending=False).reset_index(drop=True).head(20)
    return result_cosine

def find_sort_job_hybrid(resume_file, use_collaborative=True, collaborative_weight=0.3, use_experience=True):
    """
    Hybrid recommendation: Content-Based + Collaborative Filtering + Experience Analysis
    
    Args:
        resume_file: Path to resume file
        use_collaborative: Enable collaborative filtering (default: True)
        collaborative_weight: Weight cho collaborative score (0-1, default: 0.3)
        use_experience: Enable experience analysis (default: True)
    
    Returns:
        DataFrame với job recommendations (same format as find_sort_job)
        Falls back to content-based nếu collaborative filtering fails
    """
    # 1. Get content-based recommendations (EXISTING CODE - FAST)
    content_based = find_sort_job(resume_file)
    
    # 2. Try experience analysis (OPTIONAL - FAST)
    if use_experience:
        try:
            exp_analyzer = experience_analyzer.ExperienceAnalyzer()
            experience_data = exp_analyzer.extract_experience(resume_file)
            
            if experience_data.get('success'):
                # Filter and adjust jobs based on experience level
                content_based = exp_analyzer.filter_jobs_by_experience(content_based, experience_data)
        except Exception as e:
            # Silent fail - không ảnh hưởng đến hệ thống
            print(f"Experience analysis failed, continuing without it: {e}")
            pass
    
    # 3. Try collaborative filtering (NEW - OPTIONAL)
    if use_collaborative:
        try:
            # Initialize tracker and filter
            tracker = user_interactions.UserInteractionTracker()
            collab_filter = collaborative_filter.CollaborativeFilter(tracker)
            
            # Get user ID
            user_id = tracker.get_user_id_from_file(resume_file)
            
            # Track job views (implicit feedback) - sử dụng Original Link cho tracking
            for _, row in content_based.head(10).iterrows():  # Track top 10
                original_link = row.get('Original Link', row['Link'])
                tracker.track_job_view(resume_file, original_link, row['Job Title'])
            
            # Get collaborative recommendations
            collab_job_links = collab_filter.recommend_jobs_collaborative(user_id, top_n=20)
            
            if collab_job_links:
                # Add collaborative score to content-based results
                content_based['Collaborative Score'] = 0.0
                
                for idx, row in content_based.iterrows():
                    # Sử dụng Original Link để match với collaborative recommendations
                    job_link = row.get('Original Link', row['Link'])
                    if job_link in collab_job_links:
                        # Get collaborative score
                        collab_score = collab_filter.get_collaborative_score(user_id, job_link)
                        content_based.at[idx, 'Collaborative Score'] = collab_score
                
                # Combine scores (hybrid)
                content_based['Skills Match'] = pd.to_numeric(content_based['Skills Match'], errors='coerce')
                content_based['Collaborative Score'] = pd.to_numeric(content_based['Collaborative Score'], errors='coerce')
                
                # Hybrid score: Content (70%) + Collaborative (30%)
                content_weight = 1 - collaborative_weight
                content_based['Hybrid Score'] = (
                    content_based['Skills Match'] * content_weight +
                    content_based['Collaborative Score'] * 100 * collaborative_weight
                )
                
                # Sort by hybrid score
                content_based = content_based.sort_values('Hybrid Score', ascending=False).reset_index(drop=True)
                
                # Drop temporary columns
                content_based = content_based.drop(columns=['Collaborative Score', 'Hybrid Score'])
            
        except Exception as e:
            # Fallback to content-based nếu có lỗi (đảm bảo hệ thống vẫn hoạt động)
            print(f"Collaborative filtering failed, using content-based only: {e}")
            pass
    
    # Tạo link tìm kiếm Indeed từ Job Title thay vì dùng link gốc (có thể đã hết hạn)
    # Làm sau cùng để không ảnh hưởng đến collaborative filtering
    content_based['Link'] = content_based['Job Title'].apply(create_indeed_search_link)
    
    # Xóa cột Original Link trước khi trả về (không cần hiển thị)
    if 'Original Link' in content_based.columns:
        content_based = content_based.drop(columns=['Original Link'])
    
    return content_based.head(20)

def recommend_courses_for_user(resume_file, job_recommendations_df, top_n=20):
    """
    Đề xuất khóa học cho user dựa trên CV và job recommendations
    
    Args:
        resume_file: Path to resume file
        job_recommendations_df: DataFrame với job recommendations từ find_sort_job()
        top_n: Number of courses to recommend
    
    Returns:
        DataFrame với recommended courses
    """
    try:
        # Extract user skills
        user_skills_data = resparser.skill(resume_file)
        
        # Handle different return formats
        if isinstance(user_skills_data, dict):
            user_skills = user_skills_data.get('skills', [])
        elif isinstance(user_skills_data, list):
            user_skills = user_skills_data
        else:
            user_skills = []
        
        # Ensure user_skills is a list
        if not isinstance(user_skills, list):
            user_skills = []
        
        # Check if job_recommendations_df is valid
        if job_recommendations_df.empty:
            return pd.DataFrame()
        
        # Initialize course recommender
        recommender = course_recommender.CourseRecommender()
        
        # Get course recommendations
        course_recommendations = recommender.recommend_courses(
            user_skills=user_skills,
            job_recommendations_df=job_recommendations_df,
            top_n=top_n
        )
        
        return course_recommendations
    except Exception as e:
        import traceback
        print(f"Error recommending courses: {e}")
        print(traceback.format_exc())
        return pd.DataFrame()

def get_career_path_suggestions(resume_file):
    """
    Get career path suggestions cho user dựa trên CV
    
    Args:
        resume_file: Path to resume file
    
    Returns:
        Dict với career path information:
        {
            'success': bool,
            'matched_job_title': str,
            'industry': str,
            'career_path': list of roles
        }
    """
    try:
        advisor = career_path_advisor.CareerPathAdvisor()
        career_path_data = advisor.get_career_path(resume_file)
        return career_path_data
    except Exception as e:
        import traceback
        print(f"Error getting career path: {e}")
        print(traceback.format_exc())
        return {
            'success': False,
            'matched_job_title': None,
            'industry': None,
            'career_path': []
        }

def find_sort_resume(f,link):
    os.chdir(f)
    dic = {}
    for file in os.listdir():
        lsr = []
        file_path = f"{f}\\{file}"
        if file.endswith(".pdf"):
            text = resparser.convert_pdf_to_txt(file_path)
        elif file.endswith(".doc") or file.endswith(".docx"):
            text = resparser.convert_docx_to_txt(file_path)
        lsr.append(" ".join(text))
        dic[file_path] = lsr
    fy = pd.DataFrame.from_dict(dic, orient='index')
    fy.reset_index(inplace = True)
    fy.rename(columns = {'index':'link'}, inplace = True)
    fy.rename(columns = {'0':'description'}, inplace = True)
    fun = lambda x: ' '.join([word for word in x.split() if len(word)>1 and word.lower() not in (stopw)])
    fy['description'] = fy.iloc[:,1].apply(fun)
    fy['description'] = fy['description'].apply(match.preprocessing)
    fy['Resume Title'] = fy['link'].apply(lambda x: x[x.rfind("\\")+1:len(x)+1])
    results = []
    results.append(indeed_web_scraping_using_bs4.parse_job(link))
    clean_job = fun(results[0]['description'])
    clean_job = match.preprocessing(clean_job)
    test_fy = (fy['description'].values.astype('U'))
    count_matrix = match.vectorizing(clean_job, test_fy)
    matchPercentage = match.coSim(count_matrix)
    matchPercentage = pd.DataFrame(matchPercentage, columns=['Skills Match'])
    result_cosine = fy[['Resume Title','link']]
    result_cosine = result_cosine.join(matchPercentage)
    result_cosine = result_cosine.sort_values('Skills Match', ascending=False).reset_index(drop=True).head(20)
    result_cosine = result_cosine[['Resume Title', 'Skills Match', 'link']]
    return result_cosine