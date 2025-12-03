"""
User Interaction Tracking Module
Track user interactions với jobs để build collaborative filtering
"""
import json
import os
import hashlib
from datetime import datetime
from collections import defaultdict

class UserInteractionTracker:
    """Track user interactions với jobs và courses"""
    
    def __init__(self, data_file='instance/user_interactions.json'):
        self.data_file = data_file
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Tạo file nếu chưa có"""
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        if not os.path.exists(self.data_file):
            with open(self.data_file, 'w') as f:
                json.dump({}, f)
    
    def _get_user_id(self, resume_file):
        """
        Generate user ID từ resume file (hash)
        Sử dụng hash để đảm bảo privacy và consistency
        """
        try:
            # Hash filename + first few lines để unique
            with open(resume_file, 'rb') as f:
                content = f.read(1000)  # First 1000 bytes
            return hashlib.md5(content).hexdigest()[:16]
        except Exception:
            # Fallback: hash filename
            return hashlib.md5(resume_file.encode()).hexdigest()[:16]
    
    def track_job_view(self, resume_file, job_link, job_title):
        """Track khi user xem job (implicit feedback)"""
        try:
            user_id = self._get_user_id(resume_file)
            
            # Load existing data
            with open(self.data_file, 'r') as f:
                data = json.load(f)
            
            if user_id not in data:
                data[user_id] = {
                    'jobs_viewed': [],
                    'jobs_clicked': [],
                    'first_seen': datetime.now().isoformat()
                }
            
            # Add job view (avoid duplicates)
            interaction = {
                'job_link': job_link,
                'job_title': job_title,
                'timestamp': datetime.now().isoformat()
            }
            
            # Check if already exists
            existing_links = [j['job_link'] for j in data[user_id]['jobs_viewed']]
            if job_link not in existing_links:
                data[user_id]['jobs_viewed'].append(interaction)
                
                # Save
                with open(self.data_file, 'w') as f:
                    json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error tracking job view: {e}")
    
    def track_job_click(self, resume_file, job_link, job_title=None):
        """Track khi user click 'Apply' (explicit feedback)"""
        try:
            user_id = self._get_user_id(resume_file)
            
            with open(self.data_file, 'r') as f:
                data = json.load(f)
            
            if user_id not in data:
                data[user_id] = {
                    'jobs_viewed': [],
                    'jobs_clicked': [],
                    'first_seen': datetime.now().isoformat()
                }
            
            # Add job click (avoid duplicates)
            click_data = {
                'job_link': job_link,
                'timestamp': datetime.now().isoformat()
            }
            
            if job_title:
                click_data['job_title'] = job_title
            
            existing_links = [j['job_link'] for j in data[user_id]['jobs_clicked']]
            if job_link not in existing_links:
                data[user_id]['jobs_clicked'].append(click_data)
                
                # Save
                with open(self.data_file, 'w') as f:
                    json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error tracking job click: {e}")
    
    def get_user_interactions(self, user_id):
        """Get interactions của một user"""
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
            return data.get(user_id, {})
        except Exception:
            return {}
    
    def get_all_interactions(self):
        """Get tất cả interactions"""
        try:
            with open(self.data_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    
    def get_user_id_from_file(self, resume_file):
        """Get user ID từ resume file"""
        return self._get_user_id(resume_file)

