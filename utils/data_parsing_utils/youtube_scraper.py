import re
import asyncio
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from utils import log_info, log_debug, log_error


async def scrape_youtube_video_transcript(youtube_url: str) -> str:
    """
    Asynchronously retrieves the English transcript of a YouTube video given its URL. If the transcript
    is not available due to the video not having one or transcripts being disabled, logs an error message.

    :param youtube_url: The full URL of the YouTube video.
    :return: A string containing the English transcript of the video or an empty string if the transcript is unavailable.
    """
    video_id = extract_video_id(youtube_url)  # This function is synchronous

    try:
        transcript_list = await asyncio.get_event_loop().run_in_executor(None, YouTubeTranscriptApi.list_transcripts, video_id)
        transcript = transcript_list.find_transcript(["en"])
        transcript_pieces = await asyncio.get_event_loop().run_in_executor(None, transcript.fetch)
        return " ".join([t["text"] for t in transcript_pieces])
    except (NoTranscriptFound, TranscriptsDisabled):
        log_error("Transcript not available for this video.")
        return ""


def extract_video_id(youtube_url: str) -> str:
    """
    Extracts the video ID from a given YouTube URL using regular expressions. Supports standard YouTube URLs
    and shortened youtu.be URLs.

    :param youtube_url: The full URL of the YouTube video.
    :return: The extracted video ID as a string.
    :raises ValueError: If the video ID cannot be extracted from the URL.
    """
    match = re.search(r"(?<=v=)[^&#]+", youtube_url)
    match = match or re.search(r"(?<=be/)[^&#]+", youtube_url)
    if match:
        return match.group(0)
    raise ValueError(f"Could not extract video ID from URL: {youtube_url}")

