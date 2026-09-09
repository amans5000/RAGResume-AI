import re

from langchain_text_splitters import RecursiveCharacterTextSplitter


# Common resume section headings
SECTION_KEYWORDS = [
    "EDUCATION",
    "ACADEMIC BACKGROUND",

    "EXPERIENCE",
    "WORK EXPERIENCE",
    "PROFESSIONAL EXPERIENCE",
    "EMPLOYMENT EXPERIENCE",

    "INTERNSHIP",
    "INTERNSHIPS",

    "SKILLS",
    "TECHNICAL SKILLS",
    "TECHNOLOGIES",
    "TECHNICAL EXPERTISE",

    "PROJECT",
    "PROJECTS",
    "ACADEMIC PROJECTS",
    "PERSONAL PROJECTS",
    "MAJOR PROJECTS",

    "ACHIEVEMENT",
    "ACHIEVEMENTS",

    "CERTIFICATION",
    "CERTIFICATIONS",
    "CERTIFICATE",
    "CERTIFICATES",

    "PUBLICATION",
    "PUBLICATIONS",

    "RESEARCH",
    "RESEARCH EXPERIENCE",

    "POSITION OF RESPONSIBILITY",
    "POSITIONS OF RESPONSIBILITY",
    "LEADERSHIP",

    "EXTRACURRICULAR ACTIVITIES",
    "EXTRACURRICULAR",

    "COURSEWORK",
    "RELEVANT COURSEWORK",

    "SUMMARY",
    "PROFESSIONAL SUMMARY",

    "OBJECTIVE",
    "CAREER OBJECTIVE",

    "LANGUAGES",

    "INTERESTS",
    "HOBBIES",

    "ADDITIONAL INFORMATION",
]


def normalize_heading(line):
    """
    Normalize a possible section heading.
    """

    line = line.strip()

    # Remove special characters
    line = re.sub(
        r"[^A-Za-z ]",
        "",
        line
    )

    # Convert multiple spaces into one
    line = re.sub(
        r"\s+",
        " ",
        line
    )

    return line.strip().upper()


def is_section_heading(line):
    """
    Determine whether a line is likely
    to be a resume section heading.
    """

    normalized = normalize_heading(line)

    if not normalized:
        return False

    # Exact match
    if normalized in SECTION_KEYWORDS:
        return True

    # Handle headings such as:
    # "PROJECTS AND RESEARCH"
    # "TECHNICAL SKILLS AND TOOLS"

    for keyword in SECTION_KEYWORDS:

        if normalized.startswith(keyword + " "):

            # Avoid treating long sentences
            # as section headings.
            if len(normalized.split()) <= 6:
                return True

    return False


def detect_sections(text):
    """
    Detect resume sections and return
    section name + section text.
    """

    sections = []

    current_section = "GENERAL"

    current_text = []

    lines = text.splitlines()

    for line in lines:

        cleaned_line = line.strip()

        # Ignore empty lines
        if not cleaned_line:
            continue

        if is_section_heading(cleaned_line):

            # Save previous section
            if current_text:

                sections.append(
                    {
                        "section": current_section,
                        "text": "\n".join(
                            current_text
                        ).strip()
                    }
                )

            # Start new section
            current_section = normalize_heading(
                cleaned_line
            )

            current_text = []

        else:

            current_text.append(
                cleaned_line
            )

    # Save final section
    if current_text:

        sections.append(
            {
                "section": current_section,
                "text": "\n".join(
                    current_text
                ).strip()
            }
        )

    return sections


def chunk_text(
    text,
    chunk_size=800,
    chunk_overlap=150
):
    """
    Create section-aware chunks.

    Each chunk contains:
        - text
        - section
    """

    sections = detect_sections(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    final_chunks = []

    for section in sections:

        section_name = section["section"]

        section_text = section["text"]

        if not section_text.strip():
            continue

        chunks = splitter.split_text(
            section_text
        )

        for chunk in chunks:

            if not chunk.strip():
                continue

            final_chunks.append(
                {
                    "text": chunk.strip(),
                    "section": section_name
                }
            )

    return final_chunks