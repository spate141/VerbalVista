import os
import fnmatch
import requests
import subprocess
from typing import List, Dict, Optional
from urllib.parse import urlparse
from utils import log_error, log_debug, log_info


class CodeParser:
    """
    Handles cloning or updating a git repository, finding code files within it based on specified extensions,
    reading code from those files, and extracting metadata such as the repository's description from GitHub.
    """

    @staticmethod
    def find_code_files(code_dir: Optional[str] = None, allowed_file_extensions: List[str] = ['.py', '.md']) -> List[
        str]:
        """
        Identifies and returns a list of file paths within a given directory (and its subdirectories)
        that match a set of file extensions.

        :param code_dir: The directory within which to search for code files.
        :param allowed_file_extensions: A list of strings representing file extensions to include in the search.
        :return: A list of strings, each representing a full path to a matching file.
        """
        matching_files = []
        for root, dirs, files in os.walk(code_dir):
            for extension in allowed_file_extensions:
                for filename in fnmatch.filter(files, f"*{extension}"):
                    matching_files.append(os.path.join(root, filename))
        return matching_files

    def clone_or_update_repo(self, tmp_dir: Optional[str] = None, git_repo_url: Optional[str] = None) -> Dict[str, str]:
        """
        Clones a git repository to a temporary directory, or updates it if it already exists.

        :param tmp_dir: The temporary directory to clone the repository into.
        :param git_repo_url: The URL of the git repository.
        :return: A dictionary containing the path to the cloned repository, its description, and its name.
        """
        # Get the GitHub repo description
        repo_description = self.get_repo_description(git_repo_url)

        # Extract the repo name from the URL
        path = urlparse(git_repo_url).path
        repo_name = os.path.basename(path).replace('.git', '')  # Remove .git if present

        # Construct the full path to the target directory
        target_dir = os.path.join(tmp_dir, repo_name)

        if os.path.isdir(target_dir) and os.path.isdir(os.path.join(target_dir, '.git')):
            # If the repo already exists, pull to update
            cmd = ['git', '-C', target_dir, 'pull']
            action = "updated"
        else:
            # If the repo does not exist, clone it
            cmd = ['git', 'clone', git_repo_url, target_dir]
            action = "cloned"

        try:
            subprocess.run(cmd, check=True)
            log_info(f"Repository {action} successfully in {target_dir}")
        except subprocess.CalledProcessError as e:
            log_error(f"Error updating repository: {e}")
            raise e

        return target_dir, repo_description, repo_name

    @staticmethod
    def read_code(filepath: Optional[str] = None) -> str:
        """
        Reads and returns the content of a file.

        :param filepath: The path to the file.
        :return: The content of the file as a string.
        """
        with open(filepath, 'r') as f:
            return f.read()

    @staticmethod
    def get_repo_description(git_repo_url: str) -> Optional[str]:
        """
        Retrieves the description of a GitHub repository given its URL.

        :param git_repo_url: The URL of the git repository.
        :return: The description of the repository, or None if the repository does not exist or the description could not be fetched.
        """
        # Parse the provided URL
        parsed_url = urlparse(git_repo_url)

        # Remove the trailing ".git" and split the path to get the owner and repository name
        path_parts = parsed_url.path.strip("/").replace('.git', '').split("/")
        if len(path_parts) >= 2:
            owner, repo = path_parts[:2]
        else:
            print("Invalid GitHub .git URL provided.")
            return None

        # Construct the API URL and make the request
        api_url = f"https://api.github.com/repos/{owner}/{repo}"
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            return data.get('description')
        else:
            print(f"Failed to fetch repository data: {response.status_code}")
            return None

    def extract_code(self, code_filepaths: Optional[List[str]] = None, repo_description: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Reads the content from a list of file paths and assembles it into a structured format.

        :param code_filepaths: A list of file paths from which to read code.
        :param repo_description: A description of the repository, to be included with each piece of code.
        :return: A list of dictionaries, each containing a file name, its content, and a repository description.
        """
        full_documents = []
        for code_file in code_filepaths:
            code_txt = self.read_code(code_file)
            file_name = os.path.basename(code_file)
            full_documents.append({
                "file_name": file_name,
                "extracted_text": code_txt,
                "doc_description": repo_description
            })
        return full_documents

