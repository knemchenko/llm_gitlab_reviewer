import ast
import json
import requests
import re
import logging
from collections import deque
from unidiff import PatchSet, UnidiffParseError
import urllib.parse
from dotenv import load_dotenv
import os

load_dotenv()

GITLAB_TOKEN = os.getenv("GITLAB_TOKEN")
GITLAB_URL = os.getenv("GITLAB_URL")
PROJECT_ID = int(os.getenv("PROJECT_ID", 1))
MR_IID = int(os.getenv("MR_IID", 1111))
BRANCH_NAME = os.getenv("BRANCH_NAME", "P13")
OPEN_AI_BASED_API_URL = os.getenv("OPEN_AI_BASED_API_URL")
OPEN_AI_BASED_API_KEY = os.getenv("OPEN_AI_BASED_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "1.deepseek-r1:14b")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 4096 * 4))

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
MAX_DEPTH = 5
SCORE_LIMIT = 90


def count_tokens(text: str) -> int:
    words = text.split()
    return int(len(words) / 0.75)


def get_file_content(file_path: str, branch: str) -> str:
    encoded_file_path = urllib.parse.quote(file_path, safe='')
    url = f"{GITLAB_URL}/api/v4/projects/{PROJECT_ID}/repository/files/{encoded_file_path}/raw?ref={branch}"
    resp = requests.get(url, headers={"PRIVATE-TOKEN": GITLAB_TOKEN}, verify=False)
    resp.raise_for_status()
    return resp.text


def extract_method_by_lines(file_content: str, start_line: int, end_line: int) -> str:
    try:
        tree = ast.parse(file_content)
    except SyntaxError:
        return ""
    lines = file_content.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            node_start = node.lineno - 1
            node_end = max(n.lineno for n in ast.walk(node) if hasattr(n, 'lineno')) - 1
            if node_start <= end_line and node_end >= start_line:
                method_lines = lines[node_start:node_end + 1]
                return "\n".join(method_lines).strip()
    return ""


def find_dependencies(method_code: str, file_content: str, file_path: str, branch: str) -> list:
    dependencies = []
    try:
        tree = ast.parse(method_code)
    except SyntaxError:
        return dependencies
    imports = extract_imports(file_content)
    import_map = {imp.split('/')[-1].replace('.py', ''): imp for imp in imports}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                dep_code = find_local_definition(file_content, func_name)
                if dep_code:
                    dependencies.append(dep_code)
                elif func_name in import_map:
                    dep_file = import_map[func_name]
                    try:
                        dep_content = get_file_content(dep_file, branch)
                        dep_code = find_local_definition(dep_content, func_name)
                        if dep_code:
                            dependencies.append(dep_code)
                    except requests.HTTPError:
                        logging.warning("Failed to fetch dependency file: %s", dep_file)
            elif isinstance(node.func, ast.Attribute):
                attr_name = node.func.attr
                if isinstance(node.func.value, ast.Name):
                    module_name = node.func.value.id
                    if module_name in import_map:
                        dep_file = import_map[module_name]
                        try:
                            dep_content = get_file_content(dep_file, branch)
                            dep_code = find_local_definition(dep_content, attr_name)
                            if dep_code:
                                dependencies.append(dep_code)
                        except requests.HTTPError:
                            logging.warning("Failed to fetch dependency file: %s", dep_file)
    return dependencies


def find_local_definition(file_content: str, name: str) -> str:
    try:
        tree = ast.parse(file_content)
    except SyntaxError:
        return ""
    lines = file_content.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            node_start = node.lineno - 1
            node_end = max(n.lineno for n in ast.walk(node) if hasattr(n, 'lineno')) - 1
            return "\n".join(lines[node_start:node_end + 1]).strip()
    return ""


def extract_imports(source_code: str) -> list:
    imports = set()
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.replace('.', '/') + '.py')
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.replace('.', '/') + '.py')
    return list(imports)


def gather_context(file_path: str, branch: str, diff_start: int, diff_end: int) -> str:
    try:
        file_content = get_file_content(file_path, branch)
    except requests.HTTPError as e:
        logging.warning("Failed to fetch file %s: %s", file_path, e)
        return ""
    context_parts = []
    queue = deque([(extract_method_by_lines(file_content, diff_start, diff_end), 0)])
    visited = set()
    while queue:
        method_code, depth = queue.popleft()
        if depth >= MAX_DEPTH:
            break
        if not method_code or method_code in visited:
            continue
        context_parts.append(f"\n\n# Method (Depth {depth})\n{method_code}")
        visited.add(method_code)
        deps = find_dependencies(method_code, file_content, file_path, branch)
        for dep in deps:
            if dep and dep not in visited:
                queue.append((dep, depth + 1))
    return "".join(context_parts)


def parse_llm_response(response_text: str) -> dict:
    result = {"score": None, "comment": "", "tags": "", "before": "", "after": ""}
    score_match = re.search(r'SCORE:\s*(\d+)', response_text, re.IGNORECASE)
    if score_match:
        result['score'] = int(score_match.group(1))
    else:
        result['score'] = 100
    comment_match = re.search(r'<COMMENT>(.*?)</COMMENT>', response_text, re.DOTALL | re.IGNORECASE)
    if comment_match:
        result['comment'] = comment_match.group(1).strip()
    tags_match = re.search(r'<TAGS>(.*?)</TAGS>', response_text, re.DOTALL | re.IGNORECASE)
    if tags_match:
        result['tags'] = tags_match.group(1).strip()
    before_matches = re.findall(r'<BEFORE>(.*?)</BEFORE>', response_text, re.DOTALL | re.IGNORECASE)
    if before_matches:
        result['before'] = "\n\n".join(
            f"[Block {i + 1}]\n{block.strip()}" for i, block in enumerate(before_matches)) if len(
            before_matches) > 1 else before_matches[0].strip()
    after_matches = re.findall(r'<AFTER>(.*?)</AFTER>', response_text, re.DOTALL | re.IGNORECASE)
    if after_matches:
        result['after'] = "\n\n".join(
            f"[Block {i + 1}]\n{block.strip()}" for i, block in enumerate(after_matches)) if len(after_matches) > 1 else \
        after_matches[0].strip()
    logging.debug("Parsed response: %s", result)
    return result


def ask_llm_chunk_review(diff_chunk: str, context: str) -> dict:
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {OPEN_AI_BASED_API_KEY}'
    }
    lines = diff_chunk.splitlines()
    original_code = []
    modified_code = []
    for line in lines:
        if line.startswith('---') or line.startswith('+++') or line.startswith('@@'):
            continue
        elif line.startswith('-'):
            original_code.append(line[1:].strip())
        elif line.startswith('+'):
            modified_code.append(line[1:].strip())
        else:
            original_code.append(line.strip())
            modified_code.append(line.strip())
    original_text = "\n".join(original_code) if original_code else "No changes to remove"
    modified_text = "\n".join(modified_code) if modified_code else "No changes to add"
    prompt = f"""
    You are an expert code reviewer with 10 years of experience in software engineering, specializing in code quality, maintainability, and best practices for Python (PEP 8, PEP 20) and Groovy (Jenkins pipelines). Your task is to analyze the quality of code changes in a provided diff chunk and assign a score from 0 to 100, where 100 is perfect code. Follow these strict guidelines:

    1. ROLE AND EXPERTISE
       - Act as a meticulous code reviewer with deep knowledge of Python and Groovy.
       - Focus on identifying bugs, inefficiencies, security risks, readability issues, and deviations from language standards.

    2. ANALYSIS GUIDELINES
       - Carefully evaluate the changes between the original code ("Before") and the modified code ("After").
       - Use the provided file context, if available, to inform your analysis.
       - Check for logical errors, code smells, performance issues, readability and maintainability, and security vulnerabilities.
       - For trivial changes (e.g., whitespace, comments), assign a score of 95-100 without comments, <BEFORE>, <AFTER>, or <TAGS>.
       - If context is insufficient, return:
         SCORE: 10
         <COMMENT>
         Insufficient context to analyze
         </COMMENT>

    3. RESPONSE FORMAT
       - Return ONLY a plain text response in this exact format:
         SCORE: <numeric score>
         <COMMENT>
         <comment text>
         </COMMENT>
         <TAGS>
         <comma-separated tags>
         </TAGS>
         <BEFORE>
         <original code snippet (5–10 lines)>
         </BEFORE>
         <AFTER>
         <corrected code snippet>
         </AFTER>
       - If score >= {SCORE_LIMIT}, leave <COMMENT>, <TAGS>, <BEFORE>, and <AFTER> empty.
       - If score < {SCORE_LIMIT} and a fix is suggested, include the problematic code snippet in <BEFORE> and the corrected version in <AFTER>.
       - Ensure special characters in code are properly escaped.

    Here are the changes in the chunk:

    Changed lines before:
    ```
    {original_text}
    ```

    Changed lines after:
    ```
    {modified_text}
    ```

    File context:
    {context}
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }
    resp = requests.post(OPEN_AI_BASED_API_URL, headers=headers, json=payload, verify=False)
    resp.raise_for_status()
    data = resp.json()
    raw_output = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    logging.debug("Raw LLM output: %r", raw_output)
    cleaned_output = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL).strip()
    return parse_llm_response(cleaned_output)


def post_mr_comment(mr_iid: int, body: str) -> None:
    url = f"{GITLAB_URL}/api/v4/projects/{PROJECT_ID}/merge_requests/{mr_iid}/notes"
    resp = requests.post(url, headers={"PRIVATE-TOKEN": GITLAB_TOKEN}, json={"body": body}, verify=False)
    if resp.status_code == 201:
        logging.debug("General comment posted")
    else:
        logging.error("Failed to post comment: %s - %s", resp.status_code, resp.text)


def post_inline_comment(mr_iid: int, file_path: str, new_line: int, body: str, base_sha: str, head_sha: str) -> None:
    url = f"{GITLAB_URL}/api/v4/projects/{PROJECT_ID}/merge_requests/{mr_iid}/discussions"
    payload = {
        "body": body,
        "position": {
            "base_sha": base_sha,
            "head_sha": head_sha,
            "start_sha": base_sha,
            "position_type": "text",
            "new_path": file_path,
            "new_line": new_line,
        }
    }
    logging.debug("Posting inline comment: URL=%s, Payload=%s", url, json.dumps(payload, indent=2))
    resp = requests.post(url, headers={"PRIVATE-TOKEN": GITLAB_TOKEN}, json=payload, verify=False)
    if resp.status_code == 201:
        logging.debug("Inline comment posted at %s:%d", file_path, new_line)
    else:
        logging.error("Failed to post inline comment: Status=%s, Response=%s", resp.status_code, resp.text)


def get_mr_metadata(mr_iid: int) -> dict:
    url = f"{GITLAB_URL}/api/v4/projects/{PROJECT_ID}/merge_requests/{mr_iid}"
    resp = requests.get(url, headers={"PRIVATE-TOKEN": GITLAB_TOKEN}, verify=False)
    if resp.status_code != 200:
        logging.error("Failed to fetch MR metadata: %s", resp.text)
        return {}
    data = resp.json()
    return {
        "base_sha": data.get("diff_refs", {}).get("base_sha", ""),
        "head_sha": data.get("diff_refs", {}).get("head_sha", ""),
        "source_branch": data.get("source_branch", BRANCH_NAME)
    }


def build_unified_diff_from_changes(mr_iid: int) -> str:
    changes_url = f"{GITLAB_URL}/api/v4/projects/{PROJECT_ID}/merge_requests/{mr_iid}/changes"
    resp = requests.get(changes_url, headers={"PRIVATE-TOKEN": GITLAB_TOKEN}, verify=False)
    if resp.status_code != 200:
        logging.error("Failed to fetch changes: %s", resp.text)
        return ""
    data = resp.json()
    changes = data.get("changes", [])
    lines = []
    for change in changes:
        old_path = change.get("old_path", "unknown_file")
        new_path = change.get("new_path", "unknown_file")
        diff_content = change.get("diff", "")
        if not diff_content.strip():
            continue
        lines.append(f"diff --git a/{old_path} b/{new_path}")
        lines.append("index 0000000..0000001 100644")
        lines.append(f"--- a/{old_path}")
        lines.append(f"+++ b/{new_path}")
        lines.append(diff_content)
        lines.append("")
    return "\n".join(lines)


def get_mr_diff(mr_iid: int) -> str:
    return build_unified_diff_from_changes(mr_iid)


def get_overall_summary(diff_text: str) -> str:
    tokens = count_tokens(diff_text)
    if tokens > MAX_TOKENS:
        words = diff_text.split()
        allowed_words = words[:int(MAX_TOKENS * 0.75)]
        diff_text = " ".join(allowed_words) + "\n\n[Diff truncated due to token limit]"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {OPEN_AI_BASED_API_KEY}'
    }
    prompt = f"""
    You are an expert code reviewer. Provide a concise summary (2-3 sentences) describing the overall changes made in the following diff. Include key file names and the general intent of the changes. Place the summary inside a <SUMMARY> tag.

    Diff:
    ---\\n{diff_text}\\n---

    Response format:
    <SUMMARY>
    <your concise summary here>
    </SUMMARY>
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }
    resp = requests.post(OPEN_AI_BASED_API_URL, headers=headers, json=payload, verify=False)
    resp.raise_for_status()
    data = resp.json()
    raw_output = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    cleaned_output = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL).strip()
    summary_match = re.search(r'<SUMMARY>(.*?)</SUMMARY>', cleaned_output, re.DOTALL | re.IGNORECASE)
    return summary_match.group(1).strip() if summary_match else "Unable to generate summary."


def parse_diff_to_chunks(diff_text: str) -> list:
    results = []
    if not diff_text.strip():
        logging.warning("No diff text found.")
        return results
    try:
        patch = PatchSet(diff_text)
    except UnidiffParseError as e:
        logging.warning("Unidiff parse error: %s", e)
        return results
    for patched_file in patch:
        file_path = patched_file.path
        for hunk in patched_file:
            hunk_lines = []
            first_added_line = None
            current_new_line = hunk.target_start
            for line in hunk:
                prefix = "+" if line.is_added else ("-" if line.is_removed else " ")
                hunk_lines.append(prefix + line.value.rstrip("\n"))
                if line.is_added and first_added_line is None:
                    first_added_line = current_new_line
                if not line.is_removed:
                    current_new_line += 1
            hunk_text = "\n".join(hunk_lines)
            effective_new_line = first_added_line if first_added_line is not None else hunk.target_start
            results.append({
                'file_path': file_path,
                'hunk': hunk_text,
                'old_start': hunk.source_start,
                'new_start': effective_new_line,
                'new_end': current_new_line - 1
            })
    return results


def main():
    mr_metadata = get_mr_metadata(MR_IID)
    base_sha = mr_metadata.get("base_sha", "")
    head_sha = mr_metadata.get("head_sha", "")
    source_branch = mr_metadata.get("source_branch", BRANCH_NAME)
    if not base_sha or not head_sha:
        logging.error("Failed to retrieve base_sha or head_sha. Aborting.")
        return
    diff_text = get_mr_diff(MR_IID)
    overall_summary = get_overall_summary(diff_text)
    post_mr_comment(MR_IID, f"**Overall Summary:**<br>{overall_summary}")
    chunks = parse_diff_to_chunks(diff_text)
    final_summary = []
    for chunk_info in chunks:
        file_path = chunk_info['file_path']
        hunk_text = chunk_info['hunk']
        new_start_line = chunk_info['new_start']
        new_end_line = chunk_info['new_end']
        logging.debug("Analyzing chunk in file: %s at lines %d-%d", file_path, new_start_line, new_end_line)
        context = gather_context(file_path, source_branch, new_start_line, new_end_line)
        analysis_result = ask_llm_chunk_review(hunk_text, context)
        score = analysis_result.get('score', 100)
        comment = analysis_result.get('comment', '')
        tags = analysis_result.get('tags', '')
        before_code = analysis_result.get('before', '')
        after_code = analysis_result.get('after', '')
        if score < SCORE_LIMIT:
            message = (
                f"Score: {score}<br>"
                f"Comment: {comment}<br>"
                f"Tags: {tags}"
            )
            if before_code and after_code:
                message += (
                    "<br><strong>Before:</strong><br>"
                    f"```\n{before_code}\n```"
                    "<br><strong>After:</strong><br>"
                    f"```\n{after_code}\n```"
                )
            post_inline_comment(MR_IID, file_path, new_start_line, message, base_sha, head_sha)
            final_summary.append({
                'file_path': file_path,
                'score': score,
                'comment': comment,
                'tags': tags,
                'line': new_start_line
            })
    if final_summary:
        final_summary.sort(key=lambda x: x['score'])
        summary_text = "### Full Review Summary (sorted by score ascending)<br><br><br>"
        for idx, item in enumerate(final_summary, 1):
            summary_text += (f"{idx}. File: `{item['file_path']}`, line: {item['line']}<br>Score: {item['score']}<br>"
                             f"   - Tags: {item['tags']}<br>"
                             f"   - Comment: {item['comment']}<br>")
        post_mr_comment(MR_IID, summary_text)
    else:
        logging.debug(f"All chunks scored >= {SCORE_LIMIT}. No summary posted.")


if __name__ == "__main__":
    main()
