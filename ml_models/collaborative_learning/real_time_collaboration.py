"""
Real-time Collaboration Module

This module provides real-time collaborative learning features including:
- WebSocket-based real-time communication
- Live document collaboration
- Real-time group activities
- Synchronized learning sessions
- Live progress tracking

Author: LearnStyle AI Team
Version: 1.0.0
"""

import json
import asyncio
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

class CollaborationEvent(Enum):
    """Types of collaboration events"""
    USER_JOINED = "user_joined"
    USER_LEFT = "user_left"
    MESSAGE_SENT = "message_sent"
    DOCUMENT_UPDATED = "document_updated"
    ACTIVITY_STARTED = "activity_started"
    ACTIVITY_COMPLETED = "activity_completed"
    PROGRESS_UPDATED = "progress_updated"
    POLL_CREATED = "poll_created"
    POLL_RESPONDED = "poll_responded"
    BREAKOUT_ROOM_CREATED = "breakout_room_created"

class ActivityType(Enum):
    """Types of collaborative activities"""
    BRAINSTORMING = "brainstorming"
    PROBLEM_SOLVING = "problem_solving"
    PEER_REVIEW = "peer_review"
    GROUP_DISCUSSION = "group_discussion"
    COLLABORATIVE_WRITING = "collaborative_writing"
    QUIZ_COMPETITION = "quiz_competition"
    PROJECT_WORK = "project_work"

@dataclass
class CollaborationMessage:
    """Real-time collaboration message"""
    message_id: str
    user_id: str
    group_id: str
    content: str
    message_type: str
    timestamp: datetime
    metadata: Dict = None

@dataclass
class CollaborationActivity:
    """Collaborative learning activity"""
    activity_id: str
    group_id: str
    activity_type: ActivityType
    title: str
    description: str
    participants: List[str]
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: Dict = None

@dataclass
class GroupSession:
    """Real-time group session"""
    session_id: str
    group_id: str
    participants: Set[str]
    active_activity: Optional[CollaborationActivity]
    session_data: Dict
    created_at: datetime
    last_activity: datetime

class RealTimeCollaboration:
    """
    Real-time collaborative learning system
    """
    
    def __init__(self):
        """Initialize real-time collaboration system"""
        self.active_sessions: Dict[str, GroupSession] = {}
        self.user_sessions: Dict[str, str] = {}  # user_id -> session_id
        self.message_history: List[CollaborationMessage] = []
        self.activity_history: List[CollaborationActivity] = []
        
        logger.info("Real-time Collaboration system initialized")
    
    def create_group_session(self, group_id: str, participants: List[str]) -> str:
        """Create a new group session"""
        try:
            session_id = f"session_{group_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            session = GroupSession(
                session_id=session_id,
                group_id=group_id,
                participants=set(participants),
                active_activity=None,
                session_data={},
                created_at=datetime.now(),
                last_activity=datetime.now()
            )
            
            self.active_sessions[session_id] = session
            
            # Update user session mapping
            for user_id in participants:
                self.user_sessions[user_id] = session_id
            
            # Log session creation
            self._log_collaboration_event(
                CollaborationEvent.USER_JOINED,
                session_id,
                {"group_id": group_id, "participants": participants}
            )
            
            logger.info(f"Created group session {session_id} for group {group_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error creating group session: {str(e)}")
            return None
    
    def join_session(self, user_id: str, session_id: str) -> bool:
        """Join an existing group session"""
        try:
            if session_id not in self.active_sessions:
                return False
            
            session = self.active_sessions[session_id]
            session.participants.add(user_id)
            session.last_activity = datetime.now()
            
            self.user_sessions[user_id] = session_id
            
            # Log join event
            self._log_collaboration_event(
                CollaborationEvent.USER_JOINED,
                session_id,
                {"user_id": user_id}
            )
            
            logger.info(f"User {user_id} joined session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error joining session: {str(e)}")
            return False
    
    def leave_session(self, user_id: str) -> bool:
        """Leave current group session"""
        try:
            if user_id not in self.user_sessions:
                return False
            
            session_id = self.user_sessions[user_id]
            session = self.active_sessions[session_id]
            
            session.participants.discard(user_id)
            session.last_activity = datetime.now()
            
            del self.user_sessions[user_id]
            
            # Log leave event
            self._log_collaboration_event(
                CollaborationEvent.USER_LEFT,
                session_id,
                {"user_id": user_id}
            )
            
            # Clean up empty sessions
            if not session.participants:
                self._cleanup_session(session_id)
            
            logger.info(f"User {user_id} left session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error leaving session: {str(e)}")
            return False
    
    def send_message(self, user_id: str, content: str, message_type: str = "text", 
                    metadata: Dict = None) -> Optional[CollaborationMessage]:
        """Send a message in the current session"""
        try:
            if user_id not in self.user_sessions:
                return None
            
            session_id = self.user_sessions[user_id]
            session = self.active_sessions[session_id]
            
            message = CollaborationMessage(
                message_id=f"msg_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                user_id=user_id,
                group_id=session.group_id,
                content=content,
                message_type=message_type,
                timestamp=datetime.now(),
                metadata=metadata or {}
            )
            
            self.message_history.append(message)
            session.last_activity = datetime.now()
            
            # Log message event
            self._log_collaboration_event(
                CollaborationEvent.MESSAGE_SENT,
                session_id,
                {"message_id": message.message_id, "user_id": user_id}
            )
            
            logger.info(f"Message sent by {user_id} in session {session_id}")
            return message
            
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            return None
    
    def start_activity(self, user_id: str, activity_type: ActivityType, 
                      title: str, description: str) -> Optional[CollaborationActivity]:
        """Start a collaborative activity"""
        try:
            if user_id not in self.user_sessions:
                return None
            
            session_id = self.user_sessions[user_id]
            session = self.active_sessions[session_id]
            
            # End current activity if any
            if session.active_activity:
                self.end_activity(user_id)
            
            activity = CollaborationActivity(
                activity_id=f"activity_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                group_id=session.group_id,
                activity_type=activity_type,
                title=title,
                description=description,
                participants=list(session.participants),
                status="active",
                created_at=datetime.now(),
                started_at=datetime.now()
            )
            
            session.active_activity = activity
            session.last_activity = datetime.now()
            self.activity_history.append(activity)
            
            # Log activity start
            self._log_collaboration_event(
                CollaborationEvent.ACTIVITY_STARTED,
                session_id,
                {"activity_id": activity.activity_id, "activity_type": activity_type.value}
            )
            
            logger.info(f"Activity {activity.activity_id} started in session {session_id}")
            return activity
            
        except Exception as e:
            logger.error(f"Error starting activity: {str(e)}")
            return None
    
    def end_activity(self, user_id: str, results: Dict = None) -> bool:
        """End current collaborative activity"""
        try:
            if user_id not in self.user_sessions:
                return False
            
            session_id = self.user_sessions[user_id]
            session = self.active_sessions[session_id]
            
            if not session.active_activity:
                return False
            
            activity = session.active_activity
            activity.status = "completed"
            activity.completed_at = datetime.now()
            activity.results = results or {}
            
            session.active_activity = None
            session.last_activity = datetime.now()
            
            # Log activity completion
            self._log_collaboration_event(
                CollaborationEvent.ACTIVITY_COMPLETED,
                session_id,
                {"activity_id": activity.activity_id, "results": results}
            )
            
            logger.info(f"Activity {activity.activity_id} completed in session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error ending activity: {str(e)}")
            return False
    
    def update_progress(self, user_id: str, progress_data: Dict) -> bool:
        """Update user progress in current session"""
        try:
            if user_id not in self.user_sessions:
                return False
            
            session_id = self.user_sessions[user_id]
            session = self.active_sessions[session_id]
            
            # Update session data
            if 'progress' not in session.session_data:
                session.session_data['progress'] = {}
            
            session.session_data['progress'][user_id] = {
                'data': progress_data,
                'timestamp': datetime.now().isoformat()
            }
            
            session.last_activity = datetime.now()
            
            # Log progress update
            self._log_collaboration_event(
                CollaborationEvent.PROGRESS_UPDATED,
                session_id,
                {"user_id": user_id, "progress_data": progress_data}
            )
            
            logger.info(f"Progress updated for user {user_id} in session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating progress: {str(e)}")
            return False
    
    def create_poll(self, user_id: str, question: str, options: List[str], 
                   poll_type: str = "single_choice") -> Optional[Dict]:
        """Create a poll in current session"""
        try:
            if user_id not in self.user_sessions:
                return None
            
            session_id = self.user_sessions[user_id]
            session = self.active_sessions[session_id]
            
            poll = {
                'poll_id': f"poll_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                'question': question,
                'options': options,
                'poll_type': poll_type,
                'creator_id': user_id,
                'created_at': datetime.now().isoformat(),
                'responses': {},
                'status': 'active'
            }
            
            # Store poll in session data
            if 'polls' not in session.session_data:
                session.session_data['polls'] = {}
            
            session.session_data['polls'][poll['poll_id']] = poll
            session.last_activity = datetime.now()
            
            # Log poll creation
            self._log_collaboration_event(
                CollaborationEvent.POLL_CREATED,
                session_id,
                {"poll_id": poll['poll_id'], "creator_id": user_id}
            )
            
            logger.info(f"Poll {poll['poll_id']} created in session {session_id}")
            return poll
            
        except Exception as e:
            logger.error(f"Error creating poll: {str(e)}")
            return None
    
    def respond_to_poll(self, user_id: str, poll_id: str, response: str) -> bool:
        """Respond to a poll in current session"""
        try:
            if user_id not in self.user_sessions:
                return False
            
            session_id = self.user_sessions[user_id]
            session = self.active_sessions[session_id]
            
            if 'polls' not in session.session_data or poll_id not in session.session_data['polls']:
                return False
            
            poll = session.session_data['polls'][poll_id]
            poll['responses'][user_id] = response
            session.last_activity = datetime.now()
            
            # Log poll response
            self._log_collaboration_event(
                CollaborationEvent.POLL_RESPONDED,
                session_id,
                {"poll_id": poll_id, "user_id": user_id, "response": response}
            )
            
            logger.info(f"User {user_id} responded to poll {poll_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error responding to poll: {str(e)}")
            return False
    
    def get_session_messages(self, session_id: str, limit: int = 50) -> List[Dict]:
        """Get recent messages from a session"""
        try:
            if session_id not in self.active_sessions:
                return []
            
            # Filter messages for this session
            session_messages = [
                asdict(msg) for msg in self.message_history 
                if msg.group_id == self.active_sessions[session_id].group_id
            ]
            
            # Return most recent messages
            return session_messages[-limit:]
            
        except Exception as e:
            logger.error(f"Error getting session messages: {str(e)}")
            return []
    
    def get_session_activities(self, session_id: str) -> List[Dict]:
        """Get activities for a session"""
        try:
            if session_id not in self.active_sessions:
                return []
            
            group_id = self.active_sessions[session_id].group_id
            
            # Filter activities for this group
            group_activities = [
                asdict(activity) for activity in self.activity_history 
                if activity.group_id == group_id
            ]
            
            return group_activities
            
        except Exception as e:
            logger.error(f"Error getting session activities: {str(e)}")
            return []
    
    def get_session_progress(self, session_id: str) -> Dict:
        """Get progress data for a session"""
        try:
            if session_id not in self.active_sessions:
                return {}
            
            session = self.active_sessions[session_id]
            return session.session_data.get('progress', {})
            
        except Exception as e:
            logger.error(f"Error getting session progress: {str(e)}")
            return {}
    
    def get_active_sessions(self) -> List[Dict]:
        """Get list of active sessions"""
        try:
            sessions = []
            for session_id, session in self.active_sessions.items():
                session_data = {
                    'session_id': session_id,
                    'group_id': session.group_id,
                    'participants': list(session.participants),
                    'active_activity': asdict(session.active_activity) if session.active_activity else None,
                    'created_at': session.created_at.isoformat(),
                    'last_activity': session.last_activity.isoformat()
                }
                sessions.append(session_data)
            
            return sessions
            
        except Exception as e:
            logger.error(f"Error getting active sessions: {str(e)}")
            return []
    
    def _log_collaboration_event(self, event_type: CollaborationEvent, 
                               session_id: str, data: Dict):
        """Log collaboration events"""
        try:
            event = {
                'event_type': event_type.value,
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'data': data
            }
            
            # In a real implementation, this would be stored in a database
            logger.info(f"Collaboration event: {json.dumps(event)}")
            
        except Exception as e:
            logger.error(f"Error logging collaboration event: {str(e)}")
    
    def _cleanup_session(self, session_id: str):
        """Clean up empty session"""
        try:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
                logger.info(f"Cleaned up empty session {session_id}")
                
        except Exception as e:
            logger.error(f"Error cleaning up session: {str(e)}")
    
    def get_collaboration_statistics(self) -> Dict[str, int]:
        """Get collaboration statistics"""
        try:
            stats = {
                'active_sessions': len(self.active_sessions),
                'total_messages': len(self.message_history),
                'total_activities': len(self.activity_history),
                'active_users': len(self.user_sessions)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collaboration statistics: {str(e)}")
            return {
                'active_sessions': 0,
                'total_messages': 0,
                'total_activities': 0,
                'active_users': 0
            }
