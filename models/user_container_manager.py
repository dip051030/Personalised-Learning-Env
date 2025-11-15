"""
Enhanced User Container Manager - Each User Gets Own Database Container
===========================================================================

This implementation gives each user their own complete, isolated database container
including SQLite database, ChromaDB vector store, and file storage.
"""

import os
import sqlite3
import json
import shutil
import chromadb
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import hashlib
import secrets
import logging
from dataclasses import dataclass, asdict

# Import embedding model for similarity search
try:
    from models.embedding_model import embedding_model
except ImportError:
    embedding_model = None
    logging.warning("Embedding model not available - similarity search will be limited")

logger = logging.getLogger(__name__)

@dataclass
class UserContainer:
    """Represents a user's complete database container"""
    username: str
    container_path: str
    sqlite_db_path: str
    chromadb_path: str
    files_path: str
    created_at: str
    last_accessed: str

class UserContainerManager:
    """
    Manages individual database containers for each user.
    Each user gets their own SQLite DB, ChromaDB, and file storage.
    """
    
    def __init__(self, containers_root: str = "/tmp/user_databases"):
        self.containers_root = containers_root
        self.registry_path = os.path.join("/tmp/shared_auth", "user_registry.json")
        self.sessions_path = os.path.join("/tmp/shared_auth", "sessions.json")
        
        # Ensure directories exist
        os.makedirs(containers_root, exist_ok=True)
        os.makedirs("/tmp/shared_auth", exist_ok=True)
        
        # Initialize registry if it doesn't exist
        if not os.path.exists(self.registry_path):
            self._save_registry({})
        
        # Initialize sessions if it doesn't exist
        if not os.path.exists(self.sessions_path):
            self._save_sessions({})
    
    def create_user_container(self, username: str, email: str, password: str, 
                            age: int, grade: str, user_info: str = "") -> Dict[str, Any]:
        """Create a complete database container for a new user"""
        
        # Check if user already exists
        registry = self._load_registry()
        if username in registry:
            return {"success": False, "message": "Username already exists"}
        
        try:
            # Create container directory structure
            container_path = os.path.join(self.containers_root, f"{username}_container")
            sqlite_db_path = os.path.join(container_path, f"{username}.db")
            chromadb_path = os.path.join(container_path, f"{username}_chromadb")
            files_path = os.path.join(container_path, f"{username}_files")
            
            # Create directories
            os.makedirs(container_path, exist_ok=True)
            os.makedirs(chromadb_path, exist_ok=True)
            os.makedirs(files_path, exist_ok=True)
            
            # Create user's SQLite database
            self._initialize_user_sqlite_db(sqlite_db_path, username, email, password, age, grade, user_info)
            
            # Create user's ChromaDB instance
            self._initialize_user_chromadb(chromadb_path, username)
            
            # Create user container object
            container = UserContainer(
                username=username,
                container_path=container_path,
                sqlite_db_path=sqlite_db_path,
                chromadb_path=chromadb_path,
                files_path=files_path,
                created_at=datetime.now().isoformat(),
                last_accessed=datetime.now().isoformat()
            )
            
            # Register user container
            registry[username] = asdict(container)
            self._save_registry(registry)
            
            logger.info(f"Created complete database container for user: {username}")
            return {"success": True, "message": "User container created successfully"}
            
        except Exception as e:
            logger.error(f"Error creating user container for {username}: {e}")
            return {"success": False, "message": f"Container creation failed: {str(e)}"}
    
    def _initialize_user_sqlite_db(self, db_path: str, username: str, email: str, 
                                 password: str, age: int, grade: str, user_info: str):
        """Initialize SQLite database for a user"""
        
        # Hash password
        salt = secrets.token_hex(16)
        password_hash = hashlib.sha256((salt + password).encode()).hexdigest()
        hashed_password = f"{salt}:{password_hash}"
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create user profile table
        cursor.execute('''
            CREATE TABLE user_profile (
                id INTEGER PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                age INTEGER,
                grade TEXT,
                user_info TEXT,
                preferences TEXT,
                created_at TEXT,
                last_login TEXT
            )
        ''')
        
        # Create learning history table
        cursor.execute('''
            CREATE TABLE learning_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                course_id TEXT,
                topic TEXT,
                subject TEXT,
                content TEXT,
                feedback TEXT,
                study_hours INTEGER,
                grade_level TEXT,
                created_at TEXT,
                completed_at TEXT
            )
        ''')
        
        # Create user sessions table
        cursor.execute('''
            CREATE TABLE user_sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT,
                expires_at TEXT,
                last_activity TEXT,
                ip_address TEXT,
                user_agent TEXT
            )
        ''')
        
        # Create user analytics table
        cursor.execute('''
            CREATE TABLE user_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT,
                metric_value TEXT,
                recorded_at TEXT
            )
        ''')
        
        # Insert user profile
        preferences = json.dumps({
            "theme": "light",
            "language": "en", 
            "notifications": True,
            "difficulty_level": "intermediate"
        })
        
        cursor.execute('''
            INSERT INTO user_profile 
            (username, email, password_hash, age, grade, user_info, preferences, created_at, last_login)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (username, email, hashed_password, age, grade, user_info, preferences, 
              datetime.now().isoformat(), datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Initialized SQLite database for user: {username}")
    
    def _initialize_user_chromadb(self, chromadb_path: str, username: str):
        """Initialize ChromaDB instance for a user"""
        
        # Create user's personal ChromaDB client
        client = chromadb.PersistentClient(path=chromadb_path)
        
        # Create initial collections for the user
        lessons_collection = client.create_collection(
            name="lessons",
            metadata={"description": f"Personal lessons collection for {username}"}
        )
        
        scraped_collection = client.create_collection(
            name="scraped_data", 
            metadata={"description": f"Personal scraped content for {username}"}
        )
        
        courses_collection = client.create_collection(
            name="generated_courses",
            metadata={"description": f"Generated courses for {username}"}
        )
        
        logger.info(f"Initialized ChromaDB for user: {username} with {len(client.list_collections())} collections")
    
    def get_user_container(self, username: str) -> Optional[UserContainer]:
        """Get user's container information"""
        registry = self._load_registry()
        if username not in registry:
            return None
        
        container_data = registry[username]
        
        # Update last accessed time
        container_data["last_accessed"] = datetime.now().isoformat()
        registry[username] = container_data
        self._save_registry(registry)
        
        return UserContainer(**container_data)
    
    def get_user_sqlite_connection(self, username: str) -> Optional[sqlite3.Connection]:
        """Get SQLite connection for a user"""
        container = self.get_user_container(username)
        if not container:
            return None
        
        if not os.path.exists(container.sqlite_db_path):
            logger.error(f"SQLite database not found for user: {username}")
            return None
        
        return sqlite3.connect(container.sqlite_db_path)
    
    def get_user_chromadb_client(self, username: str) -> Optional[chromadb.PersistentClient]:
        """Get ChromaDB client for a user"""
        container = self.get_user_container(username)
        if not container:
            return None
        
        if not os.path.exists(container.chromadb_path):
            logger.error(f"ChromaDB not found for user: {username}")
            return None
        
        return chromadb.PersistentClient(path=container.chromadb_path)
    
    def save_user_course(self, username: str, course_data: Dict[str, Any]) -> bool:
        """Save a course to user's personal database and ChromaDB for similarity search"""
        conn = self.get_user_sqlite_connection(username)
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            course_id = course_data.get('course_id', f"course_{datetime.now().timestamp()}")
            
            cursor.execute('''
                INSERT INTO learning_history 
                (course_id, topic, subject, content, feedback, study_hours, grade_level, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                course_id,
                course_data.get('topic'),
                course_data.get('subject'),
                course_data.get('content'),
                json.dumps(course_data.get('feedback', {})),
                course_data.get('study_hours'),
                course_data.get('grade'),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            
            # Also save to ChromaDB for similarity search
            self._save_to_chromadb(username, course_id, course_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving course for user {username}: {e}")
            return False
        finally:
            conn.close()
    
    def _save_to_chromadb(self, username: str, course_id: str, course_data: Dict[str, Any]):
        """Save course content to ChromaDB for similarity search"""
        try:
            container = self.get_user_container(username)
            if not container:
                return
            
            # Initialize ChromaDB client for this user
            chroma_client = chromadb.PersistentClient(path=container.chromadb_path)
            
            # Get or create user's courses collection
            collection_name = f"{username}_courses"
            try:
                collection = chroma_client.get_collection(collection_name)
            except:
                # Create collection with embedding function if available
                embedding_function = None
                if embedding_model:
                    try:
                        embedding_function = embedding_model
                    except:
                        pass
                
                collection = chroma_client.create_collection(
                    name=collection_name,
                    embedding_function=embedding_function
                )
            
            # Prepare document content for embeddings
            content_text = f"{course_data.get('topic', '')} {course_data.get('subject', '')} {course_data.get('content', '')}"
            
            # Prepare metadata
            metadata = {
                "topic": course_data.get('topic', ''),
                "subject": course_data.get('subject', ''),
                "study_hours": str(course_data.get('study_hours', '')),
                "grade": str(course_data.get('grade', '')),
                "created_at": course_data.get('created_at', datetime.now().isoformat())
            }
            
            # Add to collection
            collection.add(
                documents=[content_text],
                metadatas=[metadata],
                ids=[course_id]
            )
            
            logger.info(f"Saved course {course_id} to ChromaDB for user {username}")
            
        except Exception as e:
            logger.error(f"Error saving to ChromaDB for user {username}: {e}")
    
    def get_user_courses(self, username: str) -> List[Dict[str, Any]]:
        """Get all courses for a user from their personal database"""
        conn = self.get_user_sqlite_connection(username)
        if not conn:
            return []
        
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT course_id, topic, subject, content, feedback, study_hours, 
                       grade_level, created_at, completed_at
                FROM learning_history 
                ORDER BY created_at DESC
            ''')
            
            courses = []
            for row in cursor.fetchall():
                courses.append({
                    'course_id': row[0],
                    'topic': row[1],
                    'subject': row[2],
                    'content': row[3],
                    'feedback': json.loads(row[4]) if row[4] else {},
                    'study_hours': row[5],
                    'grade': row[6],
                    'created_at': row[7],
                    'completed_at': row[8]
                })
            
            return courses
            
        except Exception as e:
            logger.error(f"Error getting courses for user {username}: {e}")
            return []
        finally:
            conn.close()
    
    def delete_user_container(self, username: str) -> Dict[str, Any]:
        """Completely delete a user's container (DANGEROUS!)"""
        container = self.get_user_container(username)
        if not container:
            return {"success": False, "message": "User container not found"}
        
        try:
            # Remove container directory
            if os.path.exists(container.container_path):
                shutil.rmtree(container.container_path)
            
            # Remove from registry
            registry = self._load_registry()
            if username in registry:
                del registry[username]
                self._save_registry(registry)
            
            logger.info(f"Deleted complete container for user: {username}")
            return {"success": True, "message": "User container deleted successfully"}
            
        except Exception as e:
            logger.error(f"Error deleting container for user {username}: {e}")
            return {"success": False, "message": f"Container deletion failed: {str(e)}"}
    
    def backup_user_container(self, username: str, backup_path: str) -> Dict[str, Any]:
        """Create a complete backup of user's container"""
        container = self.get_user_container(username)
        if not container:
            return {"success": False, "message": "User container not found"}
        
        try:
            backup_file = os.path.join(backup_path, f"{username}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tar.gz")
            
            # Create tar.gz backup
            import tarfile
            with tarfile.open(backup_file, "w:gz") as tar:
                tar.add(container.container_path, arcname=f"{username}_container")
            
            return {"success": True, "message": f"Backup created: {backup_file}"}
            
        except Exception as e:
            logger.error(f"Error backing up container for user {username}: {e}")
            return {"success": False, "message": f"Backup failed: {str(e)}"}
    
    def list_all_containers(self) -> List[Dict[str, Any]]:
        """List all user containers with stats"""
        registry = self._load_registry()
        containers = []
        
        for username, container_data in registry.items():
            container = UserContainer(**container_data)
            
            # Get container stats
            stats = {
                "username": username,
                "created_at": container.created_at,
                "last_accessed": container.last_accessed,
                "container_size": self._get_directory_size(container.container_path),
                "courses_count": len(self.get_user_courses(username))
            }
            containers.append(stats)
        
        return containers
    
    def get_user_learning_state(self, username: str) -> Dict[str, Any]:
        """Get user's learning state from their container"""
        container = self.get_user_container(username)
        if not container:
            raise ValueError(f"No container found for user {username}")
        
        learning_state_path = os.path.join(container.files_path, "learning_state.json")
        
        if os.path.exists(learning_state_path):
            with open(learning_state_path, 'r') as f:
                return json.load(f)
        else:
            # Return default learning state
            return {
                "user": {
                    "username": username,
                    "id": f"user_{hash(username) % 10000}",
                    "is_active": True
                },
                "history": [],
                "progress": []
            }
    
    def save_user_learning_state(self, username: str, learning_state: Any):
        """Save user's learning state to their container"""
        container = self.get_user_container(username)
        if not container:
            raise ValueError(f"No container found for user {username}")
        
        learning_state_path = os.path.join(container.files_path, "learning_state.json")
        
        # Convert to dict if it's a Pydantic model
        if hasattr(learning_state, 'model_dump'):
            state_dict = learning_state.model_dump()
        elif hasattr(learning_state, 'dict'):
            state_dict = learning_state.dict()
        else:
            state_dict = learning_state
        
        with open(learning_state_path, 'w') as f:
            json.dump(state_dict, f, indent=2)
        
        logger.info(f"Saved learning state for user {username}")
    
    def search_similar_content(self, username: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar content within user's container using vector similarity"""
        container = self.get_user_container(username)
        if not container:
            return []
        
        try:
            # Initialize ChromaDB client for this user
            chroma_client = chromadb.PersistentClient(path=container.chromadb_path)
            
            # Get or create user's courses collection
            collection_name = f"{username}_courses"
            try:
                collection = chroma_client.get_collection(collection_name)
            except:
                # No content to search
                return []
            
            # Perform similarity search
            results = collection.query(
                query_texts=[query],
                n_results=limit
            )
            
            # Format results
            similar_content = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                    distance = results['distances'][0][i] if results['distances'] and results['distances'][0] else 0
                    
                    similar_content.append({
                        "content": doc,
                        "metadata": metadata,
                        "similarity_score": 1.0 - distance,  # Convert distance to similarity
                        "id": results['ids'][0][i] if results['ids'] and results['ids'][0] else None
                    })
            
            return similar_content
            
        except Exception as e:
            logger.error(f"Error searching similar content for {username}: {e}")
            return []
    
    def find_related_content(self, username: str, subject: str = None, topic: str = None, content_type: str = None) -> List[Dict[str, Any]]:
        """Find content related to specific criteria within user's container"""
        courses = self.get_user_courses(username)
        related_content = []
        
        for course in courses:
            match = True
            
            if subject and course.get('subject', '').lower() != subject.lower():
                match = False
            
            if topic and topic.lower() not in course.get('topic', '').lower():
                match = False
            
            if content_type and course.get('content_type', '').lower() != content_type.lower():
                match = False
            
            if match:
                related_content.append(course)
        
        # Sort by creation date (most recent first)
        related_content.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return related_content
    
    def get_content_analytics(self, username: str) -> Dict[str, Any]:
        """Get analytics about user's content patterns and similarities"""
        courses = self.get_user_courses(username)
        
        if not courses:
            return {
                "total_courses": 0,
                "subjects": {},
                "topics_distribution": {},
                "content_growth": [],
                "avg_study_hours": 0
            }
        
        # Analyze content patterns
        subjects = {}
        topics = {}
        total_hours = 0
        creation_dates = []
        
        for course in courses:
            # Subject distribution
            subject = course.get('subject', 'Unknown')
            subjects[subject] = subjects.get(subject, 0) + 1
            
            # Topic distribution
            topic = course.get('topic', 'Unknown')
            topics[topic] = topics.get(topic, 0) + 1
            
            # Study hours
            hours = course.get('study_hours', 0)
            if isinstance(hours, (int, float)):
                total_hours += hours
            elif isinstance(hours, str) and hours.isdigit():
                total_hours += int(hours)
            
            # Creation dates for growth tracking
            created_at = course.get('created_at')
            if created_at:
                creation_dates.append(created_at)
        
        # Calculate content growth (courses per month)
        content_growth = []
        if creation_dates:
            from collections import defaultdict
            monthly_counts = defaultdict(int)
            
            for date_str in creation_dates:
                try:
                    date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    month_key = date_obj.strftime('%Y-%m')
                    monthly_counts[month_key] += 1
                except:
                    continue
            
            content_growth = [
                {"month": month, "count": count}
                for month, count in sorted(monthly_counts.items())
            ]
        
        return {
            "total_courses": len(courses),
            "subjects": subjects,
            "topics_distribution": topics,
            "content_growth": content_growth,
            "avg_study_hours": total_hours / len(courses) if courses else 0,
            "unique_subjects": len(subjects),
            "unique_topics": len(topics)
        }

    def _get_directory_size(self, path: str) -> int:
        """Get total size of directory in bytes"""
        total_size = 0
        if os.path.exists(path):
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    if os.path.exists(file_path):
                        total_size += os.path.getsize(file_path)
        return total_size
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load user registry"""
        try:
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_registry(self, registry: Dict[str, Any]):
        """Save user registry"""
        with open(self.registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
    
    def _save_sessions(self, sessions: Dict[str, Any]):
        """Save sessions"""
        with open(self.sessions_path, 'w') as f:
            json.dump(sessions, f, indent=2)

# Global instance
container_manager = UserContainerManager()