import os
import gradio as gr
import pickle
import re
from youtube_comment_downloader import YoutubeCommentDownloader
import matplotlib.pyplot as plt

# Load the model and vectorizer
with open('model.pkl', 'rb') as model_file, open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    model = pickle.load(model_file)
    vectorizer = pickle.load(vectorizer_file)

# Helper functions
def clean_comment(comment):
    comment = re.sub(r"[^a-zA-Z\s]", "", comment)
    comment = comment.lower()
    return comment

def fetch_youtube_comments(video_url):
    downloader = YoutubeCommentDownloader()
    comments = []
    try:
        for comment in downloader.get_comments_from_url(video_url, sort_by=0):
            comments.append(comment['text'])
    except Exception as e:
        return f"Error fetching comments: {e}", []
    return comments

def clean_youtube_url(url):
    if 'youtube.com/watch?v=' in url:
        video_id = url.split('watch?v=')[-1].split('&')[0]
    elif 'youtu.be/' in url:
        video_id = url.split('youtu.be/')[-1].split('?')[0]
    else:
        return None
    return f"https://www.youtube.com/watch?v={video_id}"

def create_pie_chart(positive_comment, negative_comment, total_comment):
    neutral_comment = total_comment - positive_comment - negative_comment

    labels = 'Positive', 'Negative', 'Neutral'
    sizes = [positive_comment, negative_comment, neutral_comment]
    colors = ['#ff9999','#66b3ff','#99ff99']
    
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',shadow=True, startangle=140)
    ax.axis('equal')

    # Ensure the directory exists
    output_dir = os.getcwd()  # Use the current working directory
    os.makedirs(output_dir, exist_ok=True)
    
    file_path = os.path.join(output_dir, 'pie_chart.png')
    plt.savefig(file_path)
    plt.close(fig)
    
    return file_path

def analyze_comments(video_link, sentiment):
    cleaned_video_link = clean_youtube_url(video_link)
    if not cleaned_video_link:
        return "Invalid YouTube video link.", None
    
    comments = fetch_youtube_comments(cleaned_video_link)
    
    if isinstance(comments, str): 
        return comments, None
    
    if not comments:
        return "No comments found for the given video link.", None

    cleaned_comments = [clean_comment(comment) for comment in comments if clean_comment(comment).strip()]
    
    if not cleaned_comments:
        return "No valid comments found after cleaning.", None

    tfidf_comments = vectorizer.transform(cleaned_comments)
    predictions = model.predict(tfidf_comments)
    
    positive_count = len([label for label in predictions if label == 1.0])
    negative_count = len([label for label in predictions if label == -1.0])
    total_comments = len(predictions)

    pie_chart_path = create_pie_chart(positive_count, negative_count, total_comments)

    if sentiment == 'Positive':
        filtered_comments = [comment for comment, label in zip(comments, predictions) if label == 1.0]
    elif sentiment == 'Negative':
        filtered_comments = [comment for comment, label in zip(comments, predictions) if label == -1.0]
    else:
        filtered_comments = [comment for comment, label in zip(comments, predictions)]

    if not filtered_comments:
        return f"No comments matching the sentiment '{sentiment}' found.", pie_chart_path

    return "\n".join(filtered_comments), pie_chart_path


# Gradio interface
interface = gr.Interface(
    fn=analyze_comments,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter YouTube Video Link", label="YouTube Video Link"),
        gr.Dropdown(choices=["Auto", "Positive", "Negative"], value="Auto", label="Sentiment")
    ],
    outputs=[
        gr.Textbox(label="Comment Analysis"),
        gr.Image(type="filepath", label="Sentiment Pie Chart")
    ],
    title="Comment Purify - Analyze Your YouTube Comments",
    description="Enter the YouTube video link and select the sentiment to analyze the comments.",
    css='div {background: url("https://cdn.discordapp.com/attachments/1163559454000816242/1244925453773574184/noah-silliman-vhInzGLpnyI-unsplash.jpg?ex=6656e2ad&is=6655912d&hm=497ebd7e75a0bb1bd6203aed5c0d8c898b4abbc808ace8bafe7f52e2a8990886&") no-repeat center center fixed;background-size: cover;}'
)

# Launch the Gradio app
interface.launch()
