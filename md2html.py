import markdown
import os


def markdown_md2html():
    with open('Readme.md', 'r') as f:
        text = f.read()
        html = markdown.markdown(text)

    with open('Readme.html', 'w') as f:
        f.write(html)


def md2pdf():

    os.system(fr"mdpdf -o {cwd}\Readme.pdf {cwd}\Readme.md")


if __name__ == '__main__':
    cwd = os.getcwd()
    markdown_md2html()
    md2pdf()
    pass
