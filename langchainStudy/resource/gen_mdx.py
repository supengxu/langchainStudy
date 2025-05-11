# 生成 .mdx 文件
file_contents = {
    "office_file.mdx": "# Microsoft Office\n",
    "markdown.mdx": "# Markdown\n",
    "json.mdx": "# JSON\n",
    "pdf.mdx": "---\n",
    "index.mdx": "---\n",
    "file_directory.mdx": "# File Directory\n",
    "csv.mdx": "# CSV\n",
    "html.mdx": "# HTML\n"
}

for filename, content in file_contents.items():
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)