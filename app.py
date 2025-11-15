import asyncio
from flask import Flask, request, jsonify, session
from flask_cors import CORS
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nodes import graph_run
from schemas import LearningState
from db.vector_db import get_user_courses
from datetime import datetime
import logging
from models.user_manager import user_manager
from models.user_container_manager import UserContainerManager
from functools import wraps

# Initialize container manager for full user data isolation
container_manager = UserContainerManager()

app = Flask(__name__)
CORS(app, supports_credentials=True)  # Enable CORS with credentials for sessions
app.secret_key = 'your-secret-key-change-this-in-production'  # Change this in production

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def require_auth(f):
    """Decorator to require authentication for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get session ID from header first, then from request body if it's a POST
        session_id = request.headers.get('Authorization')
        
        # If no header, try to get from JSON body (only for POST requests)
        if not session_id and request.method == 'POST' and request.is_json:
            try:
                data = request.get_json()
                session_id = data.get('session_id') if data else None
            except:
                pass  # Ignore JSON parsing errors
        
        if not session_id:
            return jsonify({"status": "error", "message": "Authentication required"}), 401
        
        # Remove 'Bearer ' prefix if present
        if session_id.startswith('Bearer '):
            session_id = session_id[7:]
        
        user = user_manager.validate_session(session_id)
        if not user:
            return jsonify({"status": "error", "message": "Invalid or expired session"}), 401
        
        # Add user to request context
        request.current_user = user
        return f(*args, **kwargs)
    
    return decorated_function

@app.route('/')
def home():
    return jsonify({"status": "success", "message": "Personalized Learning System API is running."})

# Authentication routes
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    
    if not data:
        return jsonify({"status": "error", "message": "No data provided"}), 400
    
    required_fields = ['username', 'email', 'password', 'age', 'grade']
    for field in required_fields:
        if not data.get(field):
            return jsonify({"status": "error", "message": f"{field} is required"}), 400
    
    result = user_manager.register_user(
        username=data['username'],
        email=data['email'],
        password=data['password'],
        age=int(data['age']),
        grade=data['grade'],
        user_info=data.get('userInfo', '')
    )
    
    if result['success']:
        # Create user container for isolated data storage
        try:
            container_manager.create_user_container(
                username=data['username'],
                email=data['email'],
                password=data['password'],
                age=int(data['age']),
                grade=data['grade'],
                user_info=data.get('userInfo', '')
            )
            logging.info(f"Created user container for {data['username']}")
        except Exception as e:
            logging.warning(f"Container creation failed for {data['username']}: {e}")
        
        return jsonify({"status": "success", "message": result['message']})
    else:
        return jsonify({"status": "error", "message": result['message']}), 400

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    
    if not data:
        return jsonify({"status": "error", "message": "No data provided"}), 400
    
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({"status": "error", "message": "Username and password are required"}), 400
    
    # Get client info
    ip_address = request.remote_addr
    user_agent = request.headers.get('User-Agent', '')
    
    result = user_manager.login_user(username, password, ip_address, user_agent)
    
    if result['success']:
        # Ensure user has a container (for existing users who registered before container system)
        try:
            container_info = container_manager.get_user_container_info(username)
        except:
            # User doesn't have a container, create one
            try:
                user_data = result['user']
                container_manager.create_user_container(
                    username=username,
                    email=user_data.get('email', ''),
                    password=password,  # We have it from login
                    age=user_data.get('age', 18),
                    grade=user_data.get('grade', '12'),
                    user_info=user_data.get('user_info', '')
                )
                logging.info(f"Created user container for existing user {username}")
            except Exception as e:
                logging.warning(f"Failed to create container for existing user {username}: {e}")
        
        return jsonify({
            "status": "success", 
            "message": result['message'],
            "session_id": result['session_id'],
            "user": result['user']
        })
    else:
        return jsonify({"status": "error", "message": result['message']}), 401

@app.route('/logout', methods=['POST'])
def logout():
    session_id = request.headers.get('Authorization') or request.json.get('session_id') if request.json else None
    
    if session_id and session_id.startswith('Bearer '):
        session_id = session_id[7:]
    
    if session_id:
        user_manager.logout_user(session_id)
    
    return jsonify({"status": "success", "message": "Logged out successfully"})

@app.route('/validate_session', methods=['GET'])
def validate_session():
    """Validate current session and return user info"""
    session_id = request.headers.get('Authorization')
    
    if not session_id:
        return jsonify({"success": False, "message": "No session token provided"}), 401
    
    # Remove 'Bearer ' prefix if present
    if session_id.startswith('Bearer '):
        session_id = session_id[7:]
    
    user = user_manager.validate_session(session_id)
    if not user:
        return jsonify({"success": False, "message": "Invalid or expired session"}), 401
    
    return jsonify({
        "success": True,
        "user": {
            "username": user.username,
            "email": user.email,
            "age": user.age,
            "grade": user.grade
        }
    })

@app.route('/profile', methods=['GET'])
@require_auth
def get_profile():
    user = request.current_user
    stats = user_manager.get_user_stats(user.username)
    
    return jsonify({
        "status": "success",
        "user": {
            "username": user.username,
            "email": user.email,
            "age": user.age,
            "grade": user.grade,
            "user_info": user.user_info,
            "preferences": user.preferences,
            "created_at": user.created_at,
            "last_login": user.last_login
        },
        "stats": stats
    })

@app.route('/update_preferences', methods=['POST'])
@require_auth
def update_preferences():
    data = request.get_json()
    
    if not data or 'preferences' not in data:
        return jsonify({"status": "error", "message": "Preferences data required"}), 400
    
    user = request.current_user
    result = user_manager.update_user_preferences(user.username, data['preferences'])
    
    if result['success']:
        return jsonify({"status": "success", "message": result['message']})
    else:
        return jsonify({"status": "error", "message": result['message']}), 400

import json

@app.route('/generate_course', methods=['POST'])
@require_auth
def generate_course():
    print("asdasd")
    data = request.get_json()
    user = request.current_user
    
    # Validate required fields
    if not data:
        return jsonify({"status": "error", "message": "No data provided"}), 400
    
    # Get user-specific data from container
    try:
        user_data = container_manager.get_user_learning_state(user.username)
        # Ensure user data has all required fields (for backwards compatibility)
        if 'user' not in user_data:
            user_data['user'] = {}
        
        user_data['user'].update({
            "username": user.username,
            "age": str(user.age),
            "grade": str(user.grade),
            "id": user_data['user'].get('id', f"user_{hash(user.username) % 10000}"),
            "is_active": True,
            "user_info": user.user_info or ""
        })
        
        # Update current resource from the form
        user_data['current_resource'] = {
            "subject": data.get('subject'),
            "grade": int(data.get('grade', user.grade)),
            "topic": data.get('topic'),
            "hours": int(data.get('studyHours', 1)),
        }
    except Exception as e:
        logging.info(f"No existing learning state for {user.username}, creating new one")
        user_data = {
            "user": {
                "username": user.username,
                "age": str(user.age),
                "grade": str(user.grade),
                "id": f"user_{hash(user.username) % 10000}",
                "is_active": True,
                "user_info": user.user_info or ""
            },
            "current_resource": {
                "subject": data.get('subject'),
                "grade": int(data.get('grade', user.grade)),
                "topic": data.get('topic'),
                "hours": int(data.get('studyHours', 1)),
            },
            "history": [],
        }
    except (ValueError, TypeError) as e:
        return jsonify({"status": "error", "message": f"Invalid data format: {str(e)}"}), 400

    logging.info(f"User {user.username} requested course generation with data: {user_data}")

    try:
        output = asyncio.run(graph_run(user_data))
        if not isinstance(output, LearningState):
            output = LearningState.model_validate(output)

        # Prepare course data for saving
        course_data = {
            "topic": output.current_resource.topic if output.current_resource else "Unknown Topic",
            "subject": data.get('subject'),
            "content": output.content.content if output.content else "No content generated",
            "feedback": output.feedback.model_dump() if output.feedback else None,
            "study_hours": data.get('studyHours', 1),
            "grade": data.get('grade', user.grade),
            "created_at": datetime.now().isoformat()
        }

        # Save course to user's container (isolated storage)
        container_manager.save_user_course(user.username, course_data)
        logging.info(f"Saved course to user container for {user.username}")

        # Also save learning state to container
        container_manager.save_user_learning_state(user.username, output)

        return jsonify({
            "status": "success", 
            "content": output.content.content if output.content else "No content generated",
            "topic": course_data["topic"]
        })

    except Exception as e:
        logging.error(f"An error occurred during course generation for user {user.username}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/get_user_history', methods=['GET'])
@require_auth
def get_history():
    user = request.current_user
    
    try:
        # Get courses from user container
        courses = container_manager.get_user_courses(user.username)
        logging.info(f"Retrieved {len(courses)} courses from user container for {user.username}")
        
        return jsonify({"status": "success", "history": courses})
    except Exception as e:
        logging.error(f"Error getting history for user {user.username}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/get_user_courses', methods=['GET'])
@require_auth
def get_courses():
    user = request.current_user
    
    try:
        # Get courses from user container
        courses = container_manager.get_user_courses(user.username)
        logging.info(f"Retrieved {len(courses)} courses from user container for {user.username}")
        
        if courses:
            return jsonify({"status": "success", "courses": {"documents": courses}})
        else:
            return jsonify({"status": "success", "courses": {"documents": []}})
    except Exception as e:
        logging.error(f"An error occurred while getting courses for user {user.username}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/search_similar', methods=['POST'])
@require_auth
def search_similar():
    """Search for similar content within user's container"""
    user = request.current_user
    data = request.get_json()
    
    if not data or 'query' not in data:
        return jsonify({"status": "error", "message": "Search query required"}), 400
    
    try:
        query = data['query']
        limit = data.get('limit', 5)
        
        # Search within user's container
        similar_content = container_manager.search_similar_content(
            user.username, 
            query, 
            limit=limit
        )
        
        return jsonify({
            "status": "success", 
            "similar_content": similar_content,
            "query": query
        })
    except Exception as e:
        logging.error(f"Error searching similar content for {user.username}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/find_related', methods=['POST'])
@require_auth
def find_related():
    """Find content related to a specific topic/subject within user's container"""
    user = request.current_user
    data = request.get_json()
    
    if not data:
        return jsonify({"status": "error", "message": "Search criteria required"}), 400
    
    try:
        subject = data.get('subject')
        topic = data.get('topic')
        content_type = data.get('content_type')  # e.g., 'lesson', 'exercise', 'project'
        
        # Search for related content in user's container
        related_content = container_manager.find_related_content(
            user.username,
            subject=subject,
            topic=topic,
            content_type=content_type
        )
        
        return jsonify({
            "status": "success", 
            "related_content": related_content
        })
    except Exception as e:
        logging.error(f"Error finding related content for {user.username}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/content_analytics', methods=['GET'])
@require_auth
def get_content_analytics():
    """Get analytics about user's content patterns and similarities"""
    user = request.current_user
    
    try:
        analytics = container_manager.get_content_analytics(user.username)
        
        return jsonify({
            "status": "success",
            "analytics": analytics
        })
    except Exception as e:
        logging.error(f"Error getting content analytics for {user.username}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/get_user_stats', methods=['GET'])
@require_auth
def get_user_stats():
    user = request.current_user
    
    try:
        # Get stats from container system
        stats = container_manager.get_user_stats(user.username)
        
        return jsonify({"status": "success", "stats": stats})
    except Exception as e:
        logging.error(f"Error getting stats for user {user.username}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/container/status', methods=['GET'])
@require_auth
def get_container_status():
    """Get user container status and information"""
    user = request.current_user
    
    try:
        container_info = container_manager.get_user_container_info(user.username)
        return jsonify({"status": "success", "container": container_info})
    except Exception as e:
        logging.error(f"Error getting container status for {user.username}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/container/backup', methods=['POST'])
@require_auth
def backup_user_container():
    """Create backup of user's container"""
    user = request.current_user
    
    try:
        backup_path = container_manager.backup_user_container(user.username)
        return jsonify({"status": "success", "backup_path": backup_path})
    except Exception as e:
        logging.error(f"Error backing up container for {user.username}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# Admin routes (for future use)
@app.route('/admin/cleanup_sessions', methods=['POST'])
def cleanup_sessions():
    """Admin endpoint to cleanup expired sessions"""
    # In production, this should require admin authentication
    user_manager.cleanup_expired_sessions()
    return jsonify({"status": "success", "message": "Expired sessions cleaned up"})





@app.route('/api/container-status', methods=['GET'])
def check_container_status():
    """Check if the current user has a valid container"""
    session_id = request.headers.get('Authorization')
    if not session_id:
        return jsonify({"error": "No session token provided"}), 401
    
    user_info = user_manager.get_user_info(session_id)
    if not user_info:
        return jsonify({"error": "Invalid session"}), 401
    
    username = user_info['username']
    
    try:
        container_info = container_manager.get_user_container_info(username)
        return jsonify({
            "status": "success",
            "has_container": True,
            "container_info": container_info
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "has_container": False,
            "error": str(e)
        })

@app.route('/api/test-auth', methods=['GET'])
@require_auth
def test_auth():
    """Simple endpoint to test authentication"""
    user = request.current_user
    return jsonify({
        "status": "success",
        "message": "Authentication working",
        "user": {
            "username": user.username,
            "email": user.email,
            "age": user.age,
            "grade": user.grade
        }
    })

# Health check endpoints
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)
