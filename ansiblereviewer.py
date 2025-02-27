import requests
import json
from typing import Dict, List

class AnsibleOllamaReviewer:
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model = "codellama:34b-instruct"
        
    def create_review_prompt(self, code: str, standards: List[str]) -> str:
        """Create a structured prompt for code review."""
        return f"""You are an expert Ansible code reviewer. Review the following Ansible code according to these specific standards:

Standards to check:
{chr(10).join(f'- {standard}' for standard in standards)}

Provide a detailed review with specific issues, their locations, and recommendations for improvement.

Ansible code to review:
```yaml
{code}
```

Format your response as:
1. Critical Issues (if any)
2. Security Concerns (if any)
3. Best Practice Violations
4. Improvement Suggestions
5. Overall Assessment
"""

    def review_code(self, code: str, standards: List[str] = None) -> Dict:
        """
        Review Ansible code using CodeLlama through Ollama
        """
        if standards is None:
            standards = [
                "Check for hardcoded credentials and sensitive data",
                "Verify task idempotency",
                "Validate privilege escalation usage",
                "Check for proper error handling",
                "Verify module usage best practices",
                "Validate YAML syntax and formatting",
                "Check for proper variable naming and usage",
                "Verify handler implementation",
                "Check for proper documentation and comments"
            ]

        prompt = self.create_review_prompt(code, standards)
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                review_text = response.json()["response"]
                
                return {
                    "status": "success",
                    "review": review_text,
                    "standards_checked": standards
                }
            else:
                return {
                    "status": "error",
                    "message": f"Ollama API error: {response.status_code}",
                    "details": response.text
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error during review: {str(e)}"
            }

    def batch_review(self, files: Dict[str, str]) -> Dict[str, Dict]:
        """
        Review multiple Ansible files
        """
        results = {}
        for filename, content in files.items():
            results[filename] = self.review_code(content)
        return results
