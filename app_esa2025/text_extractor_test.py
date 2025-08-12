import requests
from bs4 import BeautifulSoup


def extract_text_from_html(html_content):
    print("processing html content")
    soup = BeautifulSoup(html_content, "html.parser")
    print("soup")
    # Remove unwanted elements (script, style, etc.)
    for script_or_style in soup(["script", "style", "noscript"]):
        script_or_style.decompose()
        print("decomposing")

    # Extract remaining text
    print("get text")
    text = soup.get_text(separator=" ", strip=True)

    return text


response = requests.get("https://esa.org/baltimore2025/committee/")
text_content = extract_text_from_html(response.text)
print(text_content)
