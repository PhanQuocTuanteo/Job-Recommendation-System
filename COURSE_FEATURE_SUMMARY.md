# Course Recommendation Feature - Tóm tắt Implementation

## ✅ Đã hoàn thành

### 1. Module `course_recommender.py`
- ✅ Load và merge 4 course datasets (Coursera, Udemy, edX, Skillshare)
- ✅ Parse skills từ các formats khác nhau (JSON, comma-separated, text extraction)
- ✅ Extract missing skills từ job recommendations
- ✅ Match courses với missing skills
- ✅ Recommend advanced courses cho existing skills
- ✅ Ranking algorithm với multiple factors

### 2. Integration với `job.py`
- ✅ Thêm function `recommend_courses_for_user()`
- ✅ Tích hợp với job recommendation flow

### 3. Update `app.py`
- ✅ Gọi course recommendation sau khi có job recommendations
- ✅ Pass course data vào template

### 4. Update Template `employee.html`
- ✅ Hiển thị job recommendations (Top 20)
- ✅ Hiển thị course recommendations (Top 20) ngay phía dưới
- ✅ Styling với colors cho recommendation types
- ✅ Match score với color coding

---

## Cách hoạt động

### Flow:
```
1. User upload CV
   ↓
2. Extract skills từ CV
   ↓
3. Get job recommendations (20 jobs)
   ↓
4. Analyze top 5 jobs để tìm missing skills
   ↓
5. Recommend courses:
   - 10 courses cho missing skills (priority)
   - 10 courses nâng cao cho existing skills
   ↓
6. Display cả jobs và courses
```

### Algorithm:

**1. Missing Skills Extraction:**
- Lấy top 5 jobs được đề xuất
- Extract skills từ job descriptions
- Count frequency của mỗi skill
- So sánh với user skills → tìm missing skills
- Lấy top 10 missing skills

**2. Course Matching:**
- Match courses với target skills (missing hoặc existing)
- Tính skill overlap score
- Combine với rating và review count
- Formula: `Score = Skill Match (60%) + Rating (20%) + Reviews (20%)`

**3. Ranking:**
- Sort theo combined score
- Filter courses có skill match > 0
- Return top N courses

---

## Output Format

### Job Recommendations:
```
Job Title | Company | Skills Match | Link
```

### Course Recommendations:
```
Course Title | Platform | Skills Covered | Rating | Level | Duration | Link | Match Score | Recommendation Type
```

**Recommendation Types:**
- 🔴 **Missing Skills**: Courses để học skills còn thiếu
- 🟢 **Advanced Learning**: Courses nâng cao cho skills hiện có

**Match Score Colors:**
- 🟢 Green (≥50%): High match
- 🟡 Yellow (30-49%): Medium match
- 🔴 Red (<30%): Low match

---

## Files đã tạo/cập nhật

### Files mới:
- `resume_screening/course_recommender.py` - Main course recommendation module

### Files đã cập nhật:
- `resume_screening/job.py` - Thêm `recommend_courses_for_user()`
- `app.py` - Integrate course recommendations
- `templates/employee.html` - Hiển thị courses
- `resume_screening/match.py` - Fix regex warnings

---

## Testing

### Test import:
```python
from resume_screening import course_recommender
recommender = course_recommender.CourseRecommender()
```

### Test recommendation:
```python
from resume_screening import job
import pandas as pd

# Get job recommendations
job_recs = job.find_sort_job('path/to/resume.pdf')

# Get course recommendations
course_recs = job.recommend_courses_for_user('path/to/resume.pdf', job_recs)
print(course_recs)
```

---

## Lưu ý

1. **Performance**: 
   - Loading ~42K courses có thể mất vài giây lần đầu
   - Nên cache `CourseRecommender` instance hoặc load một lần khi app start

2. **Data Quality**:
   - Một số courses có thể không có skills rõ ràng
   - Skill extraction từ description có thể không perfect

3. **Missing Skills**:
   - Cần có job recommendations để extract missing skills
   - Nếu không có jobs phù hợp, sẽ không có missing skills courses

4. **Course Links**:
   - Một số datasets không có links
   - Có thể cần scrape hoặc generate links sau

---

## Next Steps (Optional)

1. **Caching**: Cache course dataset để tăng performance
2. **Better Skill Extraction**: Improve skill extraction từ descriptions
3. **Course Links**: Add logic để generate/fetch course links
4. **Filtering**: Add filters theo platform, level, duration
5. **Personalization**: Track user preferences để improve recommendations

---

## Usage Example

```python
# In Flask app
from resume_screening import job

@app.route('/employee_submit', methods=['POST'])
def employee_submit_data():
    # ... upload file ...
    path = 'instance/resume_files/{}'.format(f.filename)
    
    # Get job recommendations
    job_recs = job.find_sort_job(path)
    
    # Get course recommendations
    course_recs = job.recommend_courses_for_user(path, job_recs, top_n=20)
    
    # Render template với cả jobs và courses
    return render_template('employee.html', 
                         column_names=job_recs.columns.values,
                         row_data=list(job_recs.values.tolist()),
                         course_column_names=course_recs.columns.values,
                         course_row_data=list(course_recs.values.tolist()),
                         link_column="Link", zip=zip)
```

---

## Kết quả

✅ **Chức năng 1**: Đề xuất courses dựa trên Missing Skills - **HOÀN THÀNH**
✅ **Chức năng 2**: Đề xuất courses nâng cao cho Existing Skills - **HOÀN THÀNH**
✅ **Integration**: Tích hợp với job recommendation flow - **HOÀN THÀNH**
✅ **UI**: Hiển thị courses ngay phía dưới jobs - **HOÀN THÀNH**

Bạn có thể test ngay bây giờ bằng cách upload CV và xem kết quả!

