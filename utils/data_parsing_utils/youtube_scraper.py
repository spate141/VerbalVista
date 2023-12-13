import re
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from utils import log_info, log_debug, log_error


async def scrape_youtube_video_transcript(youtube_url: str) -> str:
    """
    Fetch the English transcript of a YouTube video given its URL.

    Args:
    youtube_url (str): URL of the YouTube video.

    Returns:
    str: The English transcript of the video.
    """
    # Extract the video ID from the URL
    video_id = await extract_video_id(youtube_url)

    # Attempt to fetch the transcript
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_transcript(["en"])
        transcript_pieces = transcript.fetch()
        return " ".join([t["text"] for t in transcript_pieces])
    except (NoTranscriptFound, TranscriptsDisabled):
        log_error("Transcript not available for this video.")
        return ""


async def extract_video_id(youtube_url: str) -> str:
    """
    Extract the video ID from a YouTube URL.

    Args:
    youtube_url (str): URL of the YouTube video.

    Returns:
    str: The extracted video ID.
    """
    match = re.search(r"(?<=v=)[^&#]+", youtube_url)
    match = match or re.search(r"(?<=be/)[^&#]+", youtube_url)
    if match:
        return match.group(0)
    raise ValueError(f"Could not extract video ID from URL: {youtube_url}")

