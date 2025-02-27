import argparse
import logging
import os
from typing import List, Optional, Dict, Any
from lib.prepare_model import prepare_model
from lib.code_review_gpt import CodeReviewer
from lib.download_repo import GitHubDownloader

class ApplicationError(Exception):
    """Base exception class for application-specific errors"""
    pass

class ConfigurationError(ApplicationError):
    """Raised when there's an error in configuration or arguments"""
    pass

class FileOperationError(ApplicationError):
    """Raised when there's an error operating on files"""
    pass

class CodeReviewApplication:
    def __init__(self):
        # Set up logging with a more detailed format
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger('CodeReviewApp')

    def setup_argparse(self) -> argparse.ArgumentParser:
        """Set up command-line argument parsing"""
        parser = argparse.ArgumentParser(
            description="Ansible Code Analyzer",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Prepare command
        prepare_parser = subparsers.add_parser(
            "prepare",
            help="Prepare the model",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        prepare_parser.add_argument(
            "--standards-file",
            default="standards.yaml",
            help="Path to the standards YAML file"
        )
        prepare_parser.add_argument(
            "--model-name",
            default="facebook/incoder-1B",
            help="Name of the base model to use"
        )
        prepare_parser.add_argument(
            "--output-dir",
            default="./prepared_model",
            help="Directory to save the prepared model"
        )

        # Review command
        review_parser = subparsers.add_parser(
            "review",
            help="Review Ansible code",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        review_parser.add_argument(
            "--model_path",
            default="./fine_tuned_model",
            help="Path to the prepared model"
        )
        review_parser.add_argument(
            "--files",
            required=True,
            nargs="+",
            help="Ansible files to review"
        )

        # Download command
        download_parser = subparsers.add_parser(
            "download",
            help="Download a GitHub repository",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        download_parser.add_argument(
            "--repo-name",
            required=True,
            help="Name of the repository (e.g., 'owner/repo')"
        )
        download_parser.add_argument(
            "--branch",
            required=True,
            help="Branch to download"
        )
        download_parser.add_argument(
            "--output-dir",
            required=True,
            help="Directory to save the downloaded files"
        )
        download_parser.add_argument(
            "--token",
            help="GitHub personal access token (can also use GITHUB_TOKEN env var)"
        )

        return parser

    def prepare_model(self, args: argparse.Namespace) -> None:
        """Handle the prepare command"""
        try:
            self.logger.info("Starting model preparation...")
            if not os.path.exists(args.standards_file):
                raise ConfigurationError(f"Standards file not found: {args.standards_file}")

            prepare_model(args.standards_file, args.model_name, args.output_dir)
            self.logger.info("Model preparation completed successfully")

        except Exception as e:
            self.logger.error(f"Error during model preparation: {str(e)}")
            raise ApplicationError(f"Model preparation failed: {str(e)}")

    def review_files(self, args: argparse.Namespace) -> Dict[str, List[Dict[str, Any]]]:
        """Handle the review command"""
        if not os.path.exists(args.model_path):
            raise ConfigurationError(f"Model path not found: {args.model_path}")

        results = {}
        reviewer = CodeReviewer(args.model_path)

        for file_path in args.files:
            try:
                if not os.path.exists(file_path):
                    raise FileOperationError(f"File not found: {file_path}")

                with open(file_path, 'r', encoding='utf-8') as file:
                    code = file.read()
                
                self.logger.info(f"Reviewing file: {file_path}")
                review_results = reviewer.review_code(code)
                results[file_path] = review_results

            except FileOperationError as e:
                self.logger.error(str(e))
                results[file_path] = []
            except Exception as e:
                self.logger.error(f"Error reviewing {file_path}: {str(e)}")
                results[file_path] = []

        return results

    def download_repo(self, args: argparse.Namespace) -> None:
        """Handle the download command"""
        try:
            github_token = args.token or os.environ.get('GITHUB_TOKEN')
            if not github_token:
                raise ConfigurationError(
                    "GitHub token not provided. Set the GITHUB_TOKEN environment variable or pass it as an argument."
                )

            downloader = GitHubDownloader(github_token)
            self.logger.info(f"Downloading repository: {args.repo_name}")
            downloader.download_repo(args.repo_name, args.branch, args.output_dir)
            self.logger.info("Repository downloaded successfully")

        except Exception as e:
            self.logger.error(f"Error downloading repository: {str(e)}")
            raise ApplicationError(f"Repository download failed: {str(e)}")

    def print_review_results(self, results: Dict[str, List[Dict[str, Any]]]) -> None:
        """Print review results in a formatted way"""
        for file_path, issues in results.items():
            print(f"\nReview for {file_path}:")
            print("-" * 40)
            
            if not issues:
                print("No issues found.")
                continue
                
            for issue in issues:
                print(f"\nStandard: {issue['standard']}")
                print(f"Line {issue['line_number']}: {issue['description']}")
                if issue.get('suggestion'):
                    print(f"Suggestion: {issue['suggestion']}")

    def run(self) -> int:
        """Main application entry point"""
        try:
            parser = self.setup_argparse()
            args = parser.parse_args()

            if not args.command:
                parser.print_help()
                return 1

            if args.command == "prepare":
                self.prepare_model(args)
            elif args.command == "review":
                results = self.review_files(args)
                self.print_review_results(results)
            elif args.command == "download":
                self.download_repo(args)
            
            return 0

        except ApplicationError as e:
            self.logger.error(str(e))
            return 1
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            return 1

def main():
    app = CodeReviewApplication()
    return app.run()

if __name__ == "__main__":
    exit(main())
