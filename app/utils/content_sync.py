"""
Content Synchronization System
Syncs published content from Content table to ContentLibrary for AI recommendations
"""

from datetime import datetime
from app import db
from app.models import Content, ContentLibrary, UserProgress

class ContentSyncManager:
    """Manages synchronization between Content and ContentLibrary tables"""
    
    @staticmethod
    def sync_published_content():
        """Sync all published content from Content to ContentLibrary"""
        try:
            # Get all published content
            published_content = Content.query.filter_by(status='published').all()
            
            synced_count = 0
            for content in published_content:
                # Check if content already exists in ContentLibrary
                existing = ContentLibrary.query.filter_by(
                    title=content.title,
                    content_type=content.content_type
                ).first()
                
                if not existing:
                    # Create new ContentLibrary entry
                    content_library_item = ContentLibrary(
                        title=content.title,
                        description=content.content[:500] + '...' if len(content.content) > 500 else content.content,
                        content_type=content.content_type,
                        style_tags=content.learning_style or 'multimodal',
                        difficulty_level=content.difficulty_level or 'beginner',
                        url_path=f'/content/{content.id}',
                        created_at=content.created_at
                    )
                    db.session.add(content_library_item)
                    synced_count += 1
                else:
                    # Update existing entry
                    existing.description = content.content[:500] + '...' if len(content.content) > 500 else content.content
                    existing.style_tags = content.learning_style or 'multimodal'
                    existing.difficulty_level = content.difficulty_level or 'beginner'
                    existing.url_path = f'/content/{content.id}'
            
            db.session.commit()
            return {
                'success': True,
                'synced_count': synced_count,
                'message': f'Successfully synced {synced_count} content items'
            }
            
        except Exception as e:
            db.session.rollback()
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to sync content'
            }
    
    @staticmethod
    def sync_single_content(content_id):
        """Sync a single content item to ContentLibrary"""
        try:
            content = Content.query.get(content_id)
            if not content:
                return {
                    'success': False,
                    'error': 'Content not found',
                    'message': 'Content item not found'
                }
            
            if content.status != 'published':
                return {
                    'success': False,
                    'error': 'Content not published',
                    'message': 'Only published content can be synced'
                }
            
            # Check if content already exists in ContentLibrary
            existing = ContentLibrary.query.filter_by(
                title=content.title,
                content_type=content.content_type
            ).first()
            
            if existing:
                # Update existing entry
                existing.description = content.content[:500] + '...' if len(content.content) > 500 else content.content
                existing.style_tags = content.learning_style or 'multimodal'
                existing.difficulty_level = content.difficulty_level or 'beginner'
                existing.url_path = f'/content/{content.id}'
                action = 'updated'
            else:
                # Create new ContentLibrary entry
                content_library_item = ContentLibrary(
                    title=content.title,
                    description=content.content[:500] + '...' if len(content.content) > 500 else content.content,
                    content_type=content.content_type,
                    style_tags=content.learning_style or 'multimodal',
                    difficulty_level=content.difficulty_level or 'beginner',
                    url_path=f'/content/{content.id}',
                    created_at=content.created_at
                )
                db.session.add(content_library_item)
                action = 'created'
            
            db.session.commit()
            return {
                'success': True,
                'action': action,
                'message': f'Content {action} in ContentLibrary successfully'
            }
            
        except Exception as e:
            db.session.rollback()
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to sync content'
            }
    
    @staticmethod
    def remove_from_content_library(content_id):
        """Remove content from ContentLibrary when unpublished or deleted"""
        try:
            content = Content.query.get(content_id)
            if not content:
                return {
                    'success': False,
                    'error': 'Content not found',
                    'message': 'Content item not found'
                }
            
            # Find corresponding ContentLibrary entry
            content_library_item = ContentLibrary.query.filter_by(
                title=content.title,
                content_type=content.content_type
            ).first()
            
            if content_library_item:
                # Check if any users have progress on this content
                user_progress = UserProgress.query.filter_by(
                    content_id=content_library_item.id
                ).count()
                
                if user_progress > 0:
                    # Archive instead of delete to preserve user progress
                    content_library_item.title = f"[ARCHIVED] {content_library_item.title}"
                    content_library_item.url_path = f'/content/archived/{content_id}'
                    action = 'archived'
                else:
                    # Safe to delete
                    db.session.delete(content_library_item)
                    action = 'deleted'
                
                db.session.commit()
                return {
                    'success': True,
                    'action': action,
                    'message': f'Content {action} from ContentLibrary successfully'
                }
            else:
                return {
                    'success': True,
                    'action': 'not_found',
                    'message': 'Content not found in ContentLibrary'
                }
                
        except Exception as e:
            db.session.rollback()
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to remove content from ContentLibrary'
            }
    
    @staticmethod
    def get_sync_status():
        """Get synchronization status between Content and ContentLibrary"""
        try:
            # Count published content
            published_count = Content.query.filter_by(status='published').count()
            
            # Count ContentLibrary items
            content_library_count = ContentLibrary.query.count()
            
            # Count synced items (by matching titles and types)
            synced_count = 0
            published_content = Content.query.filter_by(status='published').all()
            
            for content in published_content:
                existing = ContentLibrary.query.filter_by(
                    title=content.title,
                    content_type=content.content_type
                ).first()
                if existing:
                    synced_count += 1
            
            return {
                'published_content': published_count,
                'content_library_items': content_library_count,
                'synced_items': synced_count,
                'unsynced_items': published_count - synced_count,
                'sync_percentage': round((synced_count / published_count * 100), 2) if published_count > 0 else 0
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'message': 'Failed to get sync status'
            }
    
    @staticmethod
    def force_sync_all():
        """Force sync all published content, updating existing entries"""
        try:
            published_content = Content.query.filter_by(status='published').all()
            
            synced_count = 0
            updated_count = 0
            
            for content in published_content:
                # Check if content already exists in ContentLibrary
                existing = ContentLibrary.query.filter_by(
                    title=content.title,
                    content_type=content.content_type
                ).first()
                
                if existing:
                    # Update existing entry
                    existing.description = content.content[:500] + '...' if len(content.content) > 500 else content.content
                    existing.style_tags = content.learning_style or 'multimodal'
                    existing.difficulty_level = content.difficulty_level or 'beginner'
                    existing.url_path = f'/content/{content.id}'
                    updated_count += 1
                else:
                    # Create new ContentLibrary entry
                    content_library_item = ContentLibrary(
                        title=content.title,
                        description=content.content[:500] + '...' if len(content.content) > 500 else content.content,
                        content_type=content.content_type,
                        style_tags=content.learning_style or 'multimodal',
                        difficulty_level=content.difficulty_level or 'beginner',
                        url_path=f'/content/{content.id}',
                        created_at=content.created_at
                    )
                    db.session.add(content_library_item)
                    synced_count += 1
            
            db.session.commit()
            return {
                'success': True,
                'synced_count': synced_count,
                'updated_count': updated_count,
                'message': f'Successfully synced {synced_count} new items and updated {updated_count} existing items'
            }
            
        except Exception as e:
            db.session.rollback()
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to force sync content'
            }
