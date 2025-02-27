from ansiblereviewer import AnsibleOllamaReviewer
from datetime import datetime

# Initialize the reviewer
reviewer = AnsibleOllamaReviewer()

# Example Ansible playbook
ansible_code = """
- name: Install web server
  hosts: webservers
  become: yes
  tasks:
    - name: Update apt cache
      apt:
        update_cache: yes
        
    - name: Install nginx
      apt:
        name: nginx
        state: present
        
    - shell: echo "password123" > /etc/secret
      become_user: root
"""

# Define custom standards (optional)
custom_standards = [
    "No hardcoded secrets",
    "Use proper modules instead of shell/command",
    "Proper privilege escalation",
    "Idempotent tasks"
]

# Get review
result = reviewer.review_code(ansible_code, custom_standards)
file_name=f"output/code_review_{datetime.now().strftime('%Y%m%d_%H%M')}.log"
# Print review
if result["status"] == "success":
    print(result["review"])
    
    with open(file_name, mode="w", encoding="utf-8") as file:
        file.writelines(result["review"])
else:
    print(f"Error: {result['message']}")

# For batch review of multiple files
playbooks = {
    "web_server.yml": ansible_code,
    "database.yml": "another_playbook_content"
}
batch_results = reviewer.batch_review(playbooks)
