import argparse
import re
from pypdf import PdfReader

def remove_headers_footers(text, header_substring, footer_substring):
    # Split text into lines
    lines = text.split('\n')
    # Remove lines containing header or footer substrings
    cleaned_lines = []
    for line in lines:
        if header_substring and header_substring in line:
            continue
        if footer_substring and footer_substring in line:
            continue
        cleaned_lines.append(line)
    # Rejoin text
    return '\n'.join(cleaned_lines)

def normalize_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()

def remove_page_numbers(text):
    return re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

def correct_punctuation(text):
    corrections = {
        ' .': '.',
        ' ,': ',',
        ' !': '!',
        ' ?': '?',
        ' :': ':',
        ' ;': ';',
        '( ': '(',
        ' )': ')',
    }
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    return text

def format_lists(text):
    bullet_point_patterns = [r'•', r'-', r'●']  # Add other bullet characters as needed
    for pattern in bullet_point_patterns:
        text = re.sub(rf'(?m)^\s*{pattern}\s+', r'\n* ', text)
    return text

def fix_misaligned_text(text):
    # Assuming that a misaligned line does not start with a capital letter or number
    return re.sub(r'(?<!\.\n)(?<!\n\n)(?<![A-Z0-9])\n', ' ', text)

def clean_dots(extracted_text):
    # Define the regular expression pattern for finding sequences of dots
    # The pattern '[.]{2,}' will match any sequence of two or more dots
    pattern = re.compile(r'[.]{2,}')
    
    # Replace the found dot sequences with a single space
    cleaned_text = pattern.sub(' ', extracted_text)
    
    return cleaned_text


# open a pdf file and read all the text and write it to a text file
def pdf_to_text(pdf_file_path, header_substring=None, footer_substring=None, offset=0, limit=None):
    # open the pdf file
    pdf = PdfReader(pdf_file_path)
    # extract text from all pages
    all_text = []
    slice_ = pdf.pages[offset:limit] if limit else pdf.pages[offset:]
    for page in slice_:
        txt = page.extract_text()
        # replace \n with space
        while '\n' in txt:
            txt = txt.replace('\n', ' ')

        # clean up the text
        # txt = remove_headers_footers(txt, header_substring, footer_substring)
        # txt = normalize_whitespace(txt)
        # txt = remove_page_numbers(txt)
        # txt = correct_punctuation(txt)
        # txt = format_lists(txt)
        # txt = clean_dots(txt)
        # txt = fix_misaligned_text(txt)

        all_text.append(txt)

    # join all text
    text = ' '.join(all_text)
    return text


if __name__ == '__main__':
    # create an argument parser object with args same as pdf_to_text
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('pdf_file_path', help='Path to PDF file')
    arg_parser.add_argument('--header_substring', help='Substring contained in header lines to remove')
    arg_parser.add_argument('--footer_substring', help='Substring contained in footer lines to remove')
    arg_parser.add_argument('--offset', type=int, default=0, help='Offset of first page to extract')
    arg_parser.add_argument('--limit', type=int, default=None, help='Offset of first page to extract')
    # parse the args
    args = arg_parser.parse_args()
    
    # print the args
    print(args)

    # call pdf_to_text with args
    text = pdf_to_text(args.pdf_file_path, args.header_substring, args.footer_substring, args.offset, args.limit)
    txt_file = '.'.join(args.pdf_file_path.split('.')[:-1]) + '.txt'
    with open(txt_file, 'w') as f:
        f.write(text)