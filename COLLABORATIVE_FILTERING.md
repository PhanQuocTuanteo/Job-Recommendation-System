# Collaborative Filtering Implementation

## ✅ Đã hoàn thành

### 1. User Interaction Tracking (`user_interactions.py`)
- ✅ Track job views (implicit feedback)
- ✅ Track job clicks (explicit feedback)
- ✅ Generate unique user IDs từ resume file
- ✅ Lưu interactions vào JSON file

### 2. Collaborative Filtering Module (`collaborative_filter.py`)
- ✅ Build user-job interaction matrix
- ✅ Calculate user-user similarity (cosine similarity)
- ✅ Find similar users
- ✅ Get collaborative score cho jobs
- ✅ **Caching mechanism** để tối ưu performance:
  - Cache user-job matrix
  - Cache similarity matrix
  - Auto-rebuild sau 5 phút hoặc khi có thay đổi

### 3. Hybrid Recommendation (`job.py`)
- ✅ Function `find_sort_job_hybrid()` - kết hợp Content-Based + Collaborative
- ✅ **Backward compatible**: Function `find_sort_job()` giữ nguyên
- ✅ Fallback về content-based nếu collaborative filtering fails
- ✅ Weight: Content-Based (70%) + Collaborative (30%)

### 4. Integration với Flask (`app.py`)
- ✅ Route `/employee_submit` sử dụng hybrid recommendation
- ✅ Route `/track_job_click` để track explicit feedback
- ✅ Route `/get_courses` cũng sử dụng hybrid

### 5. Frontend Tracking (`employee.html`)
- ✅ Track job clicks khi user click "Apply"
- ✅ Async tracking (không block UI)
- ✅ Silent fail (không ảnh hưởng UX nếu tracking fails)

---

## Cách hoạt động

### Flow:
```
1. User upload CV
   ↓
2. Extract skills từ CV (Content-Based)
   ↓
3. Get content-based job recommendations (FAST - ~0.5s)
   ↓
4. Track job views (implicit feedback)
   ↓
5. Try collaborative filtering:
   - Build user-job matrix (cached)
   - Find similar users
   - Get collaborative scores
   ↓
6. Combine scores: Content (70%) + Collaborative (30%)
   ↓
7. Sort và return top 20 jobs
```

### Performance Optimization:
- **Caching**: User-job matrix và similarity matrix được cache
- **Lazy Loading**: Chỉ build matrix khi cần
- **Fast Fallback**: Nếu collaborative fails → dùng content-based ngay
- **Async Tracking**: Job clicks được track async, không block UI

---

## Tối ưu về thời gian

### 1. Caching Strategy
- Cache user-job matrix trong memory (global cache)
- Cache similarity matrix
- Auto-expire sau 5 phút hoặc khi có interaction mới
- Chỉ rebuild khi cần thiết

### 2. Lazy Evaluation
- Chỉ build matrix khi có đủ users (≥2 users)
- Chỉ tính similarity khi có matrix
- Skip collaborative nếu không có data

### 3. Fast Fallback
- Content-based luôn chạy trước (fast)
- Collaborative chỉ là bonus
- Nếu collaborative fails → return content-based ngay

### 4. Efficient Data Structures
- Sử dụng pandas DataFrame cho matrix operations
- Vectorized operations với numpy
- Cosine similarity với sklearn (optimized)

---

## Backward Compatibility

### ✅ Hệ thống vẫn hoạt động như cũ:
- Function `find_sort_job()` giữ nguyên 100%
- Nếu collaborative filtering fails → fallback về content-based
- Không có breaking changes

### ✅ Có thể tắt collaborative filtering:
```python
# Trong app.py, có thể set:
result_cosine = job.find_sort_job_hybrid(path, use_collaborative=False)
# Hoặc dùng function cũ:
result_cosine = job.find_sort_job(path)
```

---

## Files đã tạo/cập nhật

### Files mới:
- `resume_screening/user_interactions.py` - Track user interactions
- `resume_screening/collaborative_filter.py` - Collaborative filtering logic

### Files đã cập nhật:
- `resume_screening/job.py` - Thêm `find_sort_job_hybrid()`
- `app.py` - Sử dụng hybrid recommendation + track clicks
- `templates/employee.html` - Track job clicks

---

## Testing

### Test import:
```python
from resume_screening import user_interactions, collaborative_filter
from resume_screening import job

# Test hybrid recommendation
result = job.find_sort_job_hybrid('path/to/resume.pdf')
print(result)
```

### Test tracking:
```python
tracker = user_interactions.UserInteractionTracker()
tracker.track_job_click('resume.pdf', 'job_link', 'Job Title')
```

---

## Lưu ý

1. **Cold Start Problem**: 
   - Khi mới chạy, chưa có interaction data
   - Collaborative filtering sẽ không hoạt động
   - Hệ thống tự động fallback về content-based

2. **Data Storage**:
   - Interactions lưu trong `instance/user_interactions.json`
   - File này sẽ tự động tạo khi có interaction đầu tiên

3. **Privacy**:
   - User ID được hash từ resume content
   - Không lưu thông tin cá nhân
   - Chỉ lưu job links và timestamps

4. **Performance**:
   - Lần đầu build matrix có thể mất vài giây
   - Sau đó được cache → rất nhanh
   - Content-based luôn chạy trước → UX không bị ảnh hưởng

---

## Kết quả

✅ **Collaborative Filtering đã được tích hợp**
✅ **Hệ thống vẫn hoạt động trơn tru như cũ**
✅ **Tối ưu về mặt thời gian với caching**
✅ **Backward compatible 100%**

Hệ thống giờ có thể:
- Recommend jobs dựa trên similar users
- Học từ user behavior
- Cải thiện recommendations theo thời gian
- Vẫn hoạt động tốt ngay cả khi chưa có collaborative data

