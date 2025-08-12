from django.contrib import admin
from mongoengine import Document
from .models import Feed, FeedItem, Author


class AuthorInline(admin.TabularInline):
    """Inline admin for Author embedded documents"""
    model = Author
    extra = 1
    fields = ('name',)


@admin.register(Feed)
class FeedAdmin(admin.ModelAdmin):
    """Admin interface for Feed model"""
    list_display = ('title', 'source', 'created_at', 'updated_at')
    list_filter = ('source', 'created_at')
    search_fields = ('title', 'description', 'source')
    readonly_fields = ('id', 'created_at', 'updated_at')
    fieldsets = (
        ('Basic Information', {
            'fields': ('title', 'source_url', 'rss_feed_url', 'description')
        }),
        ('Additional Information', {
            'fields': ('icon', 'source')
        }),
        ('Metadata', {
            'fields': ('id', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )

    def get_queryset(self, request):
        """Override to work with mongoengine models"""
        return Feed.objects.all()


@admin.register(FeedItem)
class FeedItemAdmin(admin.ModelAdmin):
    """Admin interface for FeedItem model"""
    list_display = ('title', 'rss_feed_id', 'date_published', 'created_at')
    list_filter = ('date_published', 'created_at', 'rss_feed_id')
    search_fields = ('title', 'description_text', 'url')
    readonly_fields = ('id', 'created_at', 'updated_at')
    fieldsets = (
        ('Content', {
            'fields': ('title', 'url', 'description_text', 'thumbnail')
        }),
        ('Metadata', {
            'fields': ('rss_feed_id', 'date_published', 'authors')
        }),
        ('System', {
            'fields': ('id', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )

    def get_queryset(self, request):
        """Override to work with mongoengine models"""
        return FeedItem.objects.all()

    def get_feed_title(self, obj):
        """Get the title of the parent feed"""
        feed = Feed.objects(id=obj.rss_feed_id).first()
        return feed.title if feed else 'Unknown Feed'
    get_feed_title.short_description = 'Feed Title'
