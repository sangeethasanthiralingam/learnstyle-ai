"""
Collaborative Content Engine Module

This module provides collaborative content creation and sharing including:
- Real-time collaborative document editing
- Shared content libraries
- Collaborative content creation tools
- Content versioning and history
- Collaborative content review and feedback

Author: LearnStyle AI Team
Version: 1.0.0
"""

import json
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from datetime import datetime
from collections import defaultdict
import uuid

logger = logging.getLogger(__name__)

class ContentType(Enum):
    """Types of collaborative content"""
    DOCUMENT = "document"
    PRESENTATION = "presentation"
    SPREADSHEET = "spreadsheet"
    DIAGRAM = "diagram"
    MIND_MAP = "mind_map"
    QUIZ = "quiz"
    POLL = "poll"
    DISCUSSION = "discussion"

class ContentStatus(Enum):
    """Content status classifications"""
    DRAFT = "draft"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    PUBLISHED = "published"
    ARCHIVED = "archived"

class PermissionLevel(Enum):
    """Content permission levels"""
    READ_ONLY = "read_only"
    COMMENT = "comment"
    EDIT = "edit"
    ADMIN = "admin"

@dataclass
class ContentItem:
    """Collaborative content item"""
    content_id: str
    title: str
    content_type: ContentType
    content_data: Dict
    creator_id: str
    group_id: str
    status: ContentStatus
    permissions: Dict[str, PermissionLevel]  # user_id -> permission
    version: int
    created_at: datetime
    updated_at: datetime
    collaborators: Set[str]
    tags: List[str]
    metadata: Dict

@dataclass
class ContentVersion:
    """Content version history"""
    version_id: str
    content_id: str
    version_number: int
    content_data: Dict
    author_id: str
    created_at: datetime
    change_summary: str
    changes: List[Dict]

@dataclass
class ContentComment:
    """Content comment/feedback"""
    comment_id: str
    content_id: str
    author_id: str
    content: str
    position: Optional[Dict]  # For position-specific comments
    created_at: datetime
    resolved: bool
    replies: List[str]  # Comment IDs

class CollaborativeContentEngine:
    """
    Advanced collaborative content creation and management system
    """
    
    def __init__(self):
        """Initialize collaborative content engine"""
        self.content_items: Dict[str, ContentItem] = {}
        self.content_versions: Dict[str, List[ContentVersion]] = defaultdict(list)
        self.content_comments: Dict[str, List[ContentComment]] = defaultdict(list)
        self.user_content: Dict[str, Set[str]] = defaultdict(set)  # user_id -> content_ids
        self.group_content: Dict[str, Set[str]] = defaultdict(set)  # group_id -> content_ids
        
        logger.info("Collaborative Content Engine initialized")
    
    def create_content(self, title: str, content_type: ContentType, 
                      creator_id: str, group_id: str, 
                      initial_data: Dict = None) -> ContentItem:
        """Create new collaborative content"""
        try:
            content_id = str(uuid.uuid4())
            
            content_item = ContentItem(
                content_id=content_id,
                title=title,
                content_type=content_type,
                content_data=initial_data or {},
                creator_id=creator_id,
                group_id=group_id,
                status=ContentStatus.DRAFT,
                permissions={creator_id: PermissionLevel.ADMIN},
                version=1,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                collaborators={creator_id},
                tags=[],
                metadata={}
            )
            
            self.content_items[content_id] = content_item
            self.user_content[creator_id].add(content_id)
            self.group_content[group_id].add(content_id)
            
            # Create initial version
            self._create_content_version(content_item, creator_id, "Initial version")
            
            logger.info(f"Created content {content_id} by {creator_id}")
            return content_item
            
        except Exception as e:
            logger.error(f"Error creating content: {str(e)}")
            return None
    
    def update_content(self, content_id: str, user_id: str, 
                      content_data: Dict, change_summary: str = "") -> bool:
        """Update collaborative content"""
        try:
            if content_id not in self.content_items:
                return False
            
            content_item = self.content_items[content_id]
            
            # Check permissions
            if not self._has_edit_permission(content_item, user_id):
                return False
            
            # Store previous version
            previous_data = content_item.content_data.copy()
            
            # Update content
            content_item.content_data = content_data
            content_item.version += 1
            content_item.updated_at = datetime.now()
            content_item.collaborators.add(user_id)
            
            # Create new version
            self._create_content_version(content_item, user_id, change_summary)
            
            # Calculate changes
            changes = self._calculate_content_changes(previous_data, content_data)
            
            logger.info(f"Updated content {content_id} by {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating content: {str(e)}")
            return False
    
    def share_content(self, content_id: str, user_id: str, 
                     target_user_id: str, permission: PermissionLevel) -> bool:
        """Share content with another user"""
        try:
            if content_id not in self.content_items:
                return False
            
            content_item = self.content_items[content_id]
            
            # Check if user has admin permission
            if not self._has_admin_permission(content_item, user_id):
                return False
            
            # Grant permission
            content_item.permissions[target_user_id] = permission
            content_item.collaborators.add(target_user_id)
            self.user_content[target_user_id].add(content_id)
            
            logger.info(f"Shared content {content_id} with {target_user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error sharing content: {str(e)}")
            return False
    
    def add_comment(self, content_id: str, user_id: str, content: str, 
                   position: Dict = None) -> Optional[ContentComment]:
        """Add comment to content"""
        try:
            if content_id not in self.content_items:
                return None
            
            content_item = self.content_items[content_id]
            
            # Check permissions
            if not self._has_comment_permission(content_item, user_id):
                return None
            
            comment = ContentComment(
                comment_id=str(uuid.uuid4()),
                content_id=content_id,
                author_id=user_id,
                content=content,
                position=position,
                created_at=datetime.now(),
                resolved=False,
                replies=[]
            )
            
            self.content_comments[content_id].append(comment)
            
            logger.info(f"Added comment to content {content_id} by {user_id}")
            return comment
            
        except Exception as e:
            logger.error(f"Error adding comment: {str(e)}")
            return None
    
    def resolve_comment(self, content_id: str, comment_id: str, user_id: str) -> bool:
        """Resolve a content comment"""
        try:
            if content_id not in self.content_comments:
                return False
            
            comments = self.content_comments[content_id]
            comment = next((c for c in comments if c.comment_id == comment_id), None)
            
            if not comment:
                return False
            
            # Check if user has permission to resolve
            content_item = self.content_items[content_id]
            if not self._has_edit_permission(content_item, user_id):
                return False
            
            comment.resolved = True
            
            logger.info(f"Resolved comment {comment_id} in content {content_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error resolving comment: {str(e)}")
            return False
    
    def get_content_versions(self, content_id: str) -> List[ContentVersion]:
        """Get version history for content"""
        try:
            return self.content_versions.get(content_id, [])
            
        except Exception as e:
            logger.error(f"Error getting content versions: {str(e)}")
            return []
    
    def get_content_comments(self, content_id: str) -> List[ContentComment]:
        """Get comments for content"""
        try:
            return self.content_comments.get(content_id, [])
            
        except Exception as e:
            logger.error(f"Error getting content comments: {str(e)}")
            return []
    
    def get_user_content(self, user_id: str) -> List[ContentItem]:
        """Get content accessible to user"""
        try:
            content_items = []
            for content_id in self.user_content[user_id]:
                if content_id in self.content_items:
                    content_items.append(self.content_items[content_id])
            
            return content_items
            
        except Exception as e:
            logger.error(f"Error getting user content: {str(e)}")
            return []
    
    def get_group_content(self, group_id: str) -> List[ContentItem]:
        """Get content for group"""
        try:
            content_items = []
            for content_id in self.group_content[group_id]:
                if content_id in self.content_items:
                    content_items.append(self.content_items[content_id])
            
            return content_items
            
        except Exception as e:
            logger.error(f"Error getting group content: {str(e)}")
            return []
    
    def search_content(self, query: str, user_id: str, 
                      content_type: ContentType = None) -> List[ContentItem]:
        """Search content accessible to user"""
        try:
            results = []
            user_content_ids = self.user_content[user_id]
            
            for content_id in user_content_ids:
                if content_id not in self.content_items:
                    continue
                
                content_item = self.content_items[content_id]
                
                # Filter by content type if specified
                if content_type and content_item.content_type != content_type:
                    continue
                
                # Search in title and content
                if (query.lower() in content_item.title.lower() or
                    query.lower() in str(content_item.content_data).lower() or
                    any(query.lower() in tag.lower() for tag in content_item.tags)):
                    results.append(content_item)
            
            # Sort by relevance (simple implementation)
            results.sort(key=lambda x: x.updated_at, reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching content: {str(e)}")
            return []
    
    def create_content_template(self, content_type: ContentType, 
                              template_data: Dict) -> str:
        """Create content template for reuse"""
        try:
            template_id = f"template_{content_type.value}_{uuid.uuid4().hex[:8]}"
            
            # Store template (in real implementation, this would be in database)
            template = {
                'template_id': template_id,
                'content_type': content_type.value,
                'template_data': template_data,
                'created_at': datetime.now().isoformat()
            }
            
            logger.info(f"Created template {template_id}")
            return template_id
            
        except Exception as e:
            logger.error(f"Error creating content template: {str(e)}")
            return None
    
    def create_content_from_template(self, template_id: str, title: str,
                                   creator_id: str, group_id: str) -> Optional[ContentItem]:
        """Create content from template"""
        try:
            # In real implementation, this would load from database
            # For now, create basic content based on type
            content_type = ContentType.DOCUMENT  # Default
            initial_data = {"template_id": template_id}
            
            return self.create_content(title, content_type, creator_id, group_id, initial_data)
            
        except Exception as e:
            logger.error(f"Error creating content from template: {str(e)}")
            return None
    
    def _has_edit_permission(self, content_item: ContentItem, user_id: str) -> bool:
        """Check if user has edit permission"""
        permission = content_item.permissions.get(user_id, PermissionLevel.READ_ONLY)
        return permission in [PermissionLevel.EDIT, PermissionLevel.ADMIN]
    
    def _has_comment_permission(self, content_item: ContentItem, user_id: str) -> bool:
        """Check if user has comment permission"""
        permission = content_item.permissions.get(user_id, PermissionLevel.READ_ONLY)
        return permission in [PermissionLevel.COMMENT, PermissionLevel.EDIT, PermissionLevel.ADMIN]
    
    def _has_admin_permission(self, content_item: ContentItem, user_id: str) -> bool:
        """Check if user has admin permission"""
        permission = content_item.permissions.get(user_id, PermissionLevel.READ_ONLY)
        return permission == PermissionLevel.ADMIN
    
    def _create_content_version(self, content_item: ContentItem, 
                              author_id: str, change_summary: str):
        """Create content version"""
        try:
            version = ContentVersion(
                version_id=str(uuid.uuid4()),
                content_id=content_item.content_id,
                version_number=content_item.version,
                content_data=content_item.content_data.copy(),
                author_id=author_id,
                created_at=datetime.now(),
                change_summary=change_summary,
                changes=[]
            )
            
            self.content_versions[content_item.content_id].append(version)
            
        except Exception as e:
            logger.error(f"Error creating content version: {str(e)}")
    
    def _calculate_content_changes(self, old_data: Dict, new_data: Dict) -> List[Dict]:
        """Calculate changes between content versions"""
        try:
            changes = []
            
            # Simple change detection (in real implementation, use proper diff algorithm)
            for key in set(old_data.keys()) | set(new_data.keys()):
                old_value = old_data.get(key)
                new_value = new_data.get(key)
                
                if old_value != new_value:
                    changes.append({
                        'field': key,
                        'old_value': old_value,
                        'new_value': new_value,
                        'change_type': 'modified' if key in old_data and key in new_data else 'added' if key not in old_data else 'removed'
                    })
            
            return changes
            
        except Exception as e:
            logger.error(f"Error calculating content changes: {str(e)}")
            return []
    
    def get_content_statistics(self) -> Dict[str, int]:
        """Get content statistics"""
        try:
            stats = {
                'total_content': len(self.content_items),
                'total_versions': sum(len(versions) for versions in self.content_versions.values()),
                'total_comments': sum(len(comments) for comments in self.content_comments.values()),
                'active_users': len(self.user_content),
                'active_groups': len(self.group_content)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting content statistics: {str(e)}")
            return {
                'total_content': 0,
                'total_versions': 0,
                'total_comments': 0,
                'active_users': 0,
                'active_groups': 0
            }
