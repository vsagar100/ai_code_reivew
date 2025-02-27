import os
import io
import requests
import base64
import logging
import zipfile
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GitHubDownloader:
    def __init__(self, token: str):
        self.token = token
        self.headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json"
        }
        self.base_url = "https://api.github.com"
		
   # def download_repo(self, repo_url: str, token: str, download_path: str):
    def download_repo(self, repo_name: str, branch: str, output_dir: str):
        """
        Downloads a GitHub repository as a ZIP file and extracts it.

        Args:
            repo_url (str): The GitHub repository URL.
            token (str): The GitHub personal access token for authentication.
            download_path (str): The path to extract the downloaded repository.
        """
        repo_url = f"https://api.github.com/repos/{repo_name}/zipball/{branch}"
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.get(repo_url, headers=headers)
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                zip_ref.extractall(output_dir)
        else:
            raise Exception(f"Failed to download repository. Status code: {response.status_code}, Message: {response.text}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download a GitHub repository")
    parser.add_argument("--repo_name", help="Name of the repository (e.g., 'owner/repo')")
    parser.add_argument("--branch", help="Branch to download")
    parser.add_argument("--output_dir", help="Directory to save the downloaded files")
    parser.add_argument("--token", help="GitHub personal access token")

    args = parser.parse_args()

    main(args.repo_name, args.branch, args.token, args.output_dir)