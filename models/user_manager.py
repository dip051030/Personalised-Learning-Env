"""
User Management System for Personalized Learning Platform
Handles user authentication, session management, and data isolation
"""

import os
import json
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class User:
    """User data model"""
    username: str
    email: str
    password_hash: str
    age: int
    grade: str
    user_info: str
    created_at: str
    last_login: str
    is_active: bool = True
    preferences: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.preferences is None:
            self.preferences = {
                "theme": "light",
                "language": "en",
                "notifications": True,
                "difficulty_level": "intermediate"
            }

@dataclass
class UserSession:
    """User session data model"""
    session_id: str
    username: str
    created_at: str
    expires_at: str
    last_activity: str
    ip_address: str = ""
    user_agent: str = ""

class UserManager:
    """Manages user authentication, sessions, and data isolation"""
    
    def __init__(self, data_dir: str = "/tmp/user_data"):
        self.data_dir = data_dir
        self.users_file = os.path.join(data_dir, "users.json")
        self.sessions_file = os.path.join(data_dir, "sessions.json")
        self.session_duration = timedelta(hours=24)  # Sessions expire after 24 hours
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize user and session storage
        self._init_storage()
    
    def _init_storage(self):
        """Initialize storage files if they don't exist"""
        if not os.path.exists(self.users_file):
            with open(self.users_file, 'w') as f:
                json.dump({}, f)
        
        if not os.path.exists(self.sessions_file):
            with open(self.sessions_file, 'w') as f:
                json.dump({}, f)
    
    def _hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = secrets.token_hex(16)
        password_hash = hashlib.sha256((salt + password).encode()).hexdigest()
        return f"{salt}:{password_hash}"
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        try:
            salt, hash_value = password_hash.split(':')
            test_hash = hashlib.sha256((salt + password).encode()).hexdigest()
            return test_hash == hash_value
        except ValueError:
            return False
    
    def _load_users(self) -> Dict[str, User]:
        """Load users from storage"""
        try:
            with open(self.users_file, 'r') as f:
                users_data = json.load(f)
            return {
                username: User(**user_data) 
                for username, user_data in users_data.items()
            }
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_users(self, users: Dict[str, User]):
        """Save users to storage"""
        users_data = {
            username: asdict(user) 
            for username, user in users.items()
        }
        with open(self.users_file, 'w') as f:
            json.dump(users_data, f, indent=2)
    
    def _load_sessions(self) -> Dict[str, UserSession]:
        """Load sessions from storage"""
        try:
            with open(self.sessions_file, 'r') as f:
                sessions_data = json.load(f)
            return {
                session_id: UserSession(**session_data)
                for session_id, session_data in sessions_data.items()
            }
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_sessions(self, sessions: Dict[str, UserSession]):
        """Save sessions to storage"""
        sessions_data = {
            session_id: asdict(session)
            for session_id, session in sessions.items()
        }
        with open(self.sessions_file, 'w') as f:
            json.dump(sessions_data, f, indent=2)
    
    def register_user(self, username: str, email: str, password: str, 
                     age: int, grade: str, user_info: str = "") -> Dict[str, Any]:
        """Register a new user"""
        users = self._load_users()
        
        # Check if username already exists
        if username in users:
            return {"success": False, "message": "Username already exists"}
        
        # Check if email already exists
        for user in users.values():
            if user.email == email:
                return {"success": False, "message": "Email already registered"}
        
        # Create new user
        password_hash = self._hash_password(password)
        now = datetime.now().isoformat()
        
        new_user = User(
            username=username,
            email=email,
            password_hash=password_hash,
            age=age,
            grade=grade,
            user_info=user_info,
            created_at=now,
            last_login=now
        )
        
        users[username] = new_user
        self._save_users(users)
        
        # Create user data directory
        user_dir = os.path.join(self.data_dir, f"user_{username}")
        os.makedirs(user_dir, exist_ok=True)
        
        # Initialize user learning state
        self._init_user_learning_state(username)
        
        logger.info(f"New user registered: {username}")
        return {"success": True, "message": "User registered successfully"}
    
    def login_user(self, username: str, password: str, 
                  ip_address: str = "", user_agent: str = "") -> Dict[str, Any]:
        """Authenticate user and create session"""
        users = self._load_users()
        
        if username not in users:
            return {"success": False, "message": "Invalid username or password"}
        
        user = users[username]
        
        if not user.is_active:
            return {"success": False, "message": "Account is deactivated"}
        
        if not self._verify_password(password, user.password_hash):
            return {"success": False, "message": "Invalid username or password"}
        
        # Update last login
        user.last_login = datetime.now().isoformat()
        users[username] = user
        self._save_users(users)
        
        # Create session
        session_id = secrets.token_urlsafe(32)
        now = datetime.now()
        expires_at = now + self.session_duration
        
        session = UserSession(
            session_id=session_id,
            username=username,
            created_at=now.isoformat(),
            expires_at=expires_at.isoformat(),
            last_activity=now.isoformat(),
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        sessions = self._load_sessions()
        sessions[session_id] = session
        self._save_sessions(sessions)
        
        logger.info(f"User logged in: {username}")
        return {
            "success": True, 
            "message": "Login successful",
            "session_id": session_id,
            "user": {
                "username": user.username,
                "email": user.email,
                "age": user.age,
                "grade": user.grade,
                "preferences": user.preferences
            }
        }
    
    def logout_user(self, session_id: str) -> Dict[str, Any]:
        """Logout user and invalidate session"""
        sessions = self._load_sessions()
        
        if session_id in sessions:
            username = sessions[session_id].username
            del sessions[session_id]
            self._save_sessions(sessions)
            logger.info(f"User logged out: {username}")
            return {"success": True, "message": "Logout successful"}
        
        return {"success": False, "message": "Invalid session"}
    
    def validate_session(self, session_id: str) -> Optional[User]:
        """Validate session and return user if valid"""
        sessions = self._load_sessions()
        
        if session_id not in sessions:
            return None
        
        session = sessions[session_id]
        
        # Check if session expired
        expires_at = datetime.fromisoformat(session.expires_at)
        if datetime.now() > expires_at:
            # Remove expired session
            del sessions[session_id]
            self._save_sessions(sessions)
            return None
        
        # Update last activity
        session.last_activity = datetime.now().isoformat()
        sessions[session_id] = session
        self._save_sessions(sessions)
        
        # Get user data
        users = self._load_users()
        return users.get(session.username)
    
    def get_user_data_path(self, username: str) -> str:
        """Get user-specific data directory path"""
        return os.path.join(self.data_dir, f"user_{username}")
    
    def get_user_learning_state_path(self, username: str) -> str:
        """Get user-specific learning state file path"""
        return os.path.join(self.get_user_data_path(username), "learning_state.json")
    
    def _init_user_learning_state(self, username: str):
        """Initialize user learning state file"""
        user_dir = self.get_user_data_path(username)
        state_file = os.path.join(user_dir, "learning_state.json")
        
        if not os.path.exists(state_file):
            initial_state = {
                "user": {
                    "username": username,
                    "id": f"user_{hash(username) % 10000}",
                    "is_active": True
                },
                "current_resource": None,
                "content": None,
                "feedback": None,
                "history": [],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            with open(state_file, 'w') as f:
                json.dump(initial_state, f, indent=2)
    
    def get_user_courses(self, username: str) -> List[Dict[str, Any]]:
        """Get all courses for a specific user"""
        try:
            state_file = self.get_user_learning_state_path(username)
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    user_state = json.load(f)
                return user_state.get("history", [])
            return []
        except Exception as e:
            logger.error(f"Error getting user courses for {username}: {e}")
            return []
    
    def save_user_course(self, username: str, course_data: Dict[str, Any]):
        """Save a course for a specific user"""
        try:
            state_file = self.get_user_learning_state_path(username)
            
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    user_state = json.load(f)
            else:
                self._init_user_learning_state(username)
                with open(state_file, 'r') as f:
                    user_state = json.load(f)
            
            # Add course to history
            course_data["timestamp"] = datetime.now().isoformat()
            user_state["history"].append(course_data)
            user_state["updated_at"] = datetime.now().isoformat()
            
            with open(state_file, 'w') as f:
                json.dump(user_state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving course for {username}: {e}")
            raise
    
    def update_user_preferences(self, username: str, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Update user preferences"""
        users = self._load_users()
        
        if username not in users:
            return {"success": False, "message": "User not found"}
        
        user = users[username]
        user.preferences.update(preferences)
        users[username] = user
        self._save_users(users)
        
        return {"success": True, "message": "Preferences updated"}
    
    def get_user_stats(self, username: str) -> Dict[str, Any]:
        """Get user statistics"""
        courses = self.get_user_courses(username)
        
        total_courses = len(courses)
        total_words = sum(len(course.get("content", "").split()) for course in courses)
        subjects = list(set(course.get("subject", "unknown") for course in courses))
        
        # Calculate average rating
        ratings = [course.get("feedback", {}).get("rating", 0) for course in courses if course.get("feedback")]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        
        return {
            "total_courses": total_courses,
            "total_words": total_words,
            "subjects_explored": len(subjects),
            "average_rating": round(avg_rating, 1),
            "subjects": subjects,
            "recent_activity": courses[-5:] if courses else []
        }
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        sessions = self._load_sessions()
        now = datetime.now()
        
        expired_sessions = []
        for session_id, session in sessions.items():
            expires_at = datetime.fromisoformat(session.expires_at)
            if now > expires_at:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del sessions[session_id]
        
        if expired_sessions:
            self._save_sessions(sessions)
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

# Global user manager instance
user_manager = UserManager()