import os
from flask import Flask, render_template, redirect, request, jsonify, send_from_directory
from resume_screening import job
from resume_screening import user_interactions
from os import listdir

# import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('universal_tagset')
# nltk.download('maxent_ne_chunker')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('brown')

from nltk.corpus import stopwords

stopw  = set(stopwords.words('english'))

app=Flask(__name__)

# Add zip function to Jinja2 globals
app.jinja_env.globals.update(zip=zip)

os.makedirs(os.path.join(app.instance_path, 'resume_files'), exist_ok=True)

# Serve static files
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/img/<path:path>')
def serve_img(path):
    return send_from_directory('static/img', path)

@app.route('/styles/<path:path>')
def serve_styles(path):
    return send_from_directory('static/styles', path)

# Routes
@app.route('/employee_submit',methods=['POST'])
def employee_submit_data():
    if request.method == 'POST':        
        f=request.files['userfile']
        f.save(os.path.join(app.instance_path, 'resume_files', f.filename))
        
    path = 'instance/resume_files/{}'.format(f.filename)
    
    # Get job recommendations using hybrid approach (Content-Based + Collaborative + Experience)
    # Falls back to content-based nếu collaborative filtering fails
    result_cosine = job.find_sort_job_hybrid(path, use_collaborative=True, collaborative_weight=0.3, use_experience=True)
    
    # Find Link column index
    column_names = result_cosine.columns.values.tolist()
    link_column = None
    for idx, col in enumerate(column_names):
        if col.lower() in ['link', 'url', 'job link', 'apply link']:
            link_column = col
            break
    
    # Render HTML template with job recommendations
    return render_template('employee.html',
                         column_names=column_names,
                         row_data=result_cosine.values.tolist(),
                         link_column=link_column,
                         resume_filename=f.filename)

@app.route('/get_courses',methods=['POST'])
def get_courses():
    """API endpoint để load courses recommendations (AJAX call)"""
    if request.method == 'POST':
        resume_filename = request.form.get('resume_filename')
        if not resume_filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        path = 'instance/resume_files/{}'.format(resume_filename)
        
        try:
            # Get job recommendations để extract missing skills (use hybrid)
            job_recommendations = job.find_sort_job_hybrid(path, use_collaborative=True, collaborative_weight=0.3)
            
            # Get course recommendations
            course_recommendations = job.recommend_courses_for_user(path, job_recommendations, top_n=20)
            
            # Prepare course data
            if not course_recommendations.empty:
                course_column_names = course_recommendations.columns.values.tolist()
                course_row_data = course_recommendations.values.tolist()
                return jsonify({
                    'success': True,
                    'column_names': course_column_names,
                    'row_data': course_row_data
                })
            else:
                return jsonify({
                    'success': True,
                    'column_names': [],
                    'row_data': []
                })
        except Exception as e:
            import traceback
            return jsonify({
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }), 500

@app.route('/get_career_path',methods=['POST'])
def get_career_path():
    """API endpoint để load career path recommendations (AJAX call)"""
    if request.method == 'POST':
        resume_filename = request.form.get('resume_filename')
        if not resume_filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        path = 'instance/resume_files/{}'.format(resume_filename)
        
        try:
            # Get career path suggestions
            career_path_data = job.get_career_path_suggestions(path)
            
            return jsonify({
                'success': career_path_data.get('success', False),
                'matched_job_title': career_path_data.get('matched_job_title'),
                'industry': career_path_data.get('industry'),
                'career_path': career_path_data.get('career_path', [])
            })
        except Exception as e:
            import traceback
            return jsonify({
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }), 500

@app.route('/track_job_click',methods=['POST'])
def track_job_click():
    """Track khi user click Apply button (explicit feedback)"""
    if request.method == 'POST':
        try:
            resume_filename = request.form.get('resume_filename')
            job_link = request.form.get('job_link')
            job_title = request.form.get('job_title', '')
            
            if not resume_filename or not job_link:
                return jsonify({'success': False, 'error': 'Missing parameters'}), 400
            
            path = 'instance/resume_files/{}'.format(resume_filename)
            
            # Track click
            tracker = user_interactions.UserInteractionTracker()
            tracker.track_job_click(path, job_link, job_title)
            
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/employee")
def employee():
    return render_template("employee.html")

@app.route("/home")
def home():
    return redirect('/')

if __name__ =="__main__":
    app.run(debug=True)
