import os
import fnmatch
import requests
import subprocess
from typing import List, Dict
from urllib.parse import urlparse
from utils import log_error, log_debug, log_info


class CodeParser:
    """
    This class will clone the repo, go through all files and extract the code and save it into
    respective FILENAME.EXTENSION.data.txt file in `data/documents/REPO_NAME` directory.
    """

    @staticmethod
    def find_code_files(code_dir: str = None, allowed_file_extensions: List[str] = ('.py', '.md')) -> List[str]:
        """
        Fetch all code file paths from the given directory that match the given file extensions.
        :param code_dir:
        :param allowed_file_extensions: A list of file extensions to match.
        :return list of str: A list of paths to files that match the given extensions.
        """
        matching_files = []
        for root, dirs, files in os.walk(code_dir):
            for extension in allowed_file_extensions:
                for filename in fnmatch.filter(files, f"*{extension}"):
                    matching_files.append(os.path.join(root, filename))
        return matching_files

    def clone_or_update_repo(self, tmp_dir: str = None, git_repo_url: str = None):
        """
        Given the GitHub repo URL, this will clone the repo locally into a subdirectory
        within self.code_dir named after the repo, unless it already exists, in which case
        it updates the repo with 'git pull'.
        :param tmp_dir: Tmp directory
        :param git_repo_url: Repo URL
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
    def read_code(filepath: str = None) -> str:
        """
        Read the code from file and return full text as string.
        :param filepath: Path of the file.
        :return: file content as string format
        """
        with open(filepath, 'r') as f:
            return f.read()

    @staticmethod
    def get_repo_description(git_repo_url):
        """
        Fetches the description of a GitHub repository given its .git URL.

        Parameters:
        - git_repo_url (str): The full .git URL of the GitHub repository.

        Returns:
        - str: The repository description, or None if not found.
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

    def extract_code(self, code_filepaths: List[str] = None, repo_description: str = None) -> List[Dict[str, str]]:
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

