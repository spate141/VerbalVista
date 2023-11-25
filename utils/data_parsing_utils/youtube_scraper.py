from langchain.document_loaders import YoutubeLoader
from utils.logging_module import log_info, log_debug, log_error


def scrape_youtube_video_transcript(url):
    try:
        loader = YoutubeLoader.from_youtube_url(
            url, add_video_info=True
        )
        documents = loader.load()
        log_debug(f'Total YT documents found: {len(documents)}')
        transcript = []
        for d in documents:
            transcript.append(d.page_content)
        transcript = ' '.join(transcript)
        return transcript
    except Exception as e:
        log_error(f'Invalid YouTube URL: {e}')
        return None

