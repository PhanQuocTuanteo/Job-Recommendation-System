"""
Collaborative Filtering Module cho Job Recommendations
Tối ưu với caching để giảm thời gian tính toán
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import time

# Global cache để tránh rebuild matrix mỗi lần
_collaborative_cache = {
    'user_job_matrix': None,
    'user_similarity_matrix': None,
    'last_update': None,
    'interaction_count': 0
}

class CollaborativeFilter:
    """Collaborative Filtering cho job recommendations với caching"""
    
    def __init__(self, interaction_tracker=None, cache_duration=300):
        """
        Args:
            interaction_tracker: UserInteractionTracker instance
            cache_duration: Cache duration in seconds (default 5 minutes)
        """
        from . import user_interactions
        self.interaction_tracker = interaction_tracker or user_interactions.UserInteractionTracker()
        self.cache_duration = cache_duration
        self.user_job_matrix = None
        self.user_similarity_matrix = None
    
    def _should_rebuild_cache(self):
        """Check if cache needs to be rebuilt"""
        global _collaborative_cache
        
        if _collaborative_cache['user_job_matrix'] is None:
            return True
        
        if _collaborative_cache['last_update'] is None:
            return True
        
        # Check if cache expired
        elapsed = time.time() - _collaborative_cache['last_update']
        if elapsed > self.cache_duration:
            return True
        
        return False
    
    def build_user_job_matrix(self, force_rebuild=False):
        """
        Build user-job interaction matrix với caching
        
        Args:
            force_rebuild: Force rebuild even if cache exists
        
        Returns:
            DataFrame với user-job matrix
        """
        global _collaborative_cache
        
        # Check cache
        if not force_rebuild and not self._should_rebuild_cache():
            self.user_job_matrix = _collaborative_cache['user_job_matrix']
            self.user_similarity_matrix = _collaborative_cache['user_similarity_matrix']
            return self.user_job_matrix
        
        all_interactions = self.interaction_tracker.get_all_interactions()
        
        if not all_interactions:
            _collaborative_cache['user_job_matrix'] = None
            return None
        
        # Collect all jobs
        all_jobs = set()
        for user_data in all_interactions.values():
            # Use clicked jobs (explicit feedback) - stronger signal
            for job in user_data.get('jobs_clicked', []):
                all_jobs.add(job['job_link'])
            # Also include viewed jobs (implicit feedback) - weaker signal
            for job in user_data.get('jobs_viewed', []):
                all_jobs.add(job['job_link'])
        
        if len(all_jobs) == 0:
            _collaborative_cache['user_job_matrix'] = None
            return None
        
        # Build matrix: users x jobs
        user_ids = list(all_interactions.keys())
        job_links = list(all_jobs)
        
        matrix = np.zeros((len(user_ids), len(job_links)))
        
        for i, user_id in enumerate(user_ids):
            user_data = all_interactions[user_id]
            
            # Clicked jobs = 2.0 (strong signal)
            clicked_jobs = [j['job_link'] for j in user_data.get('jobs_clicked', [])]
            for job_link in clicked_jobs:
                if job_link in job_links:
                    j = job_links.index(job_link)
                    matrix[i][j] = 2.0
            
            # Viewed jobs = 1.0 (weak signal)
            viewed_jobs = [j['job_link'] for j in user_data.get('jobs_viewed', [])]
            for job_link in viewed_jobs:
                if job_link in job_links:
                    j = job_links.index(job_link)
                    if matrix[i][j] == 0:  # Only if not already clicked
                        matrix[i][j] = 1.0
        
        self.user_job_matrix = pd.DataFrame(
            matrix, 
            index=user_ids, 
            columns=job_links
        )
        
        # Update cache
        _collaborative_cache['user_job_matrix'] = self.user_job_matrix
        _collaborative_cache['last_update'] = time.time()
        _collaborative_cache['interaction_count'] = len(all_interactions)
        
        return self.user_job_matrix
    
    def calculate_user_similarity(self, force_rebuild=False):
        """
        Calculate user-user similarity với caching
        
        Args:
            force_rebuild: Force rebuild even if cache exists
        
        Returns:
            DataFrame với user similarity matrix
        """
        global _collaborative_cache
        
        # Check cache
        if not force_rebuild and not self._should_rebuild_cache():
            if _collaborative_cache['user_similarity_matrix'] is not None:
                self.user_similarity_matrix = _collaborative_cache['user_similarity_matrix']
                return self.user_similarity_matrix
        
        if self.user_job_matrix is None:
            self.build_user_job_matrix(force_rebuild=force_rebuild)
        
        if self.user_job_matrix is None or len(self.user_job_matrix) < 2:
            _collaborative_cache['user_similarity_matrix'] = None
            return None
        
        # Calculate cosine similarity between users
        similarity = cosine_similarity(self.user_job_matrix.values)
        self.user_similarity_matrix = pd.DataFrame(
            similarity,
            index=self.user_job_matrix.index,
            columns=self.user_job_matrix.index
        )
        
        # Update cache
        _collaborative_cache['user_similarity_matrix'] = self.user_similarity_matrix
        
        return self.user_similarity_matrix
    
    def find_similar_users(self, user_id, top_k=5):
        """
        Find k most similar users
        
        Args:
            user_id: User ID
            top_k: Number of similar users to find
        
        Returns:
            List of similar user IDs
        """
        if self.user_similarity_matrix is None:
            self.calculate_user_similarity()
        
        if self.user_similarity_matrix is None:
            return []
        
        if user_id not in self.user_similarity_matrix.index:
            return []
        
        # Get similarity scores
        similarities = self.user_similarity_matrix.loc[user_id].sort_values(ascending=False)
        # Remove self
        similarities = similarities[similarities.index != user_id]
        # Filter positive similarities only
        similarities = similarities[similarities > 0]
        
        return similarities.head(top_k).index.tolist()
    
    def get_collaborative_score(self, user_id, job_link):
        """
        Get collaborative filtering score cho một job
        
        Args:
            user_id: User ID
            job_link: Job link
        
        Returns:
            Score từ 0-1 (0 = no collaborative data, 1 = highly recommended)
        """
        if self.user_job_matrix is None:
            self.build_user_job_matrix()
        
        if self.user_job_matrix is None:
            return 0.0
        
        if user_id not in self.user_job_matrix.index:
            return 0.0
        
        if job_link not in self.user_job_matrix.columns:
            return 0.0
        
        # Find similar users
        similar_users = self.find_similar_users(user_id, top_k=5)
        
        if not similar_users:
            return 0.0
        
        # Calculate weighted score
        total_score = 0.0
        total_weight = 0.0
        
        for similar_user_id in similar_users:
            similarity = self.user_similarity_matrix.loc[user_id, similar_user_id]
            
            # Check if this user interacted with this job
            if self.user_job_matrix.loc[similar_user_id, job_link] > 0:
                # Weight by similarity and interaction strength
                interaction_strength = self.user_job_matrix.loc[similar_user_id, job_link]
                total_score += similarity * interaction_strength
                total_weight += 1
        
        if total_weight == 0:
            return 0.0
        
        # Normalize to 0-1 (max interaction strength is 2.0)
        normalized_score = min(total_score / (len(similar_users) * 2.0), 1.0)
        
        return normalized_score
    
    def recommend_jobs_collaborative(self, user_id, top_n=10):
        """
        Recommend jobs dựa trên similar users
        
        Args:
            user_id: User ID
            top_n: Number of jobs to recommend
        
        Returns:
            List of job links recommended
        """
        if self.user_job_matrix is None:
            self.build_user_job_matrix()
        
        if self.user_job_matrix is None or user_id not in self.user_job_matrix.index:
            return []
        
        # Find similar users
        similar_users = self.find_similar_users(user_id, top_k=5)
        
        if not similar_users:
            return []
        
        # Get jobs interacted by similar users
        job_scores = defaultdict(float)
        
        for similar_user_id in similar_users:
            similarity_score = self.user_similarity_matrix.loc[user_id, similar_user_id]
            
            # Get jobs interacted by this similar user
            user_jobs = self.user_job_matrix.loc[similar_user_id]
            interacted_jobs = user_jobs[user_jobs > 0]
            
            # Add scores (weighted by similarity and interaction strength)
            for job_link, interaction_strength in interacted_jobs.items():
                job_scores[job_link] += similarity_score * interaction_strength
        
        # Sort by score
        recommended_jobs = sorted(job_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [job_link for job_link, score in recommended_jobs[:top_n]]

