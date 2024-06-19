from youtube_comment_downloader import YoutubeCommentDownloader

def fetch_youtube_comments(video_url):
    downloader = YoutubeCommentDownloader()
    comments = []
    try:
        for comment in downloader.get_comments_from_url(video_url, sort_by=0):  # Use 0 for 'top comments'
            comments.append(comment['text'])
    except Exception as e:
        print(f"Error fetching comments: {e}")
        return []
    return comments

video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
comments = fetch_youtube_comments(video_url)
print(f"Total comments fetched: {len(comments)}")
print(comments[:5])  # Print first 5 comments for inspection
