import re
import subprocess
from argparse import ArgumentParser
from datetime import date
from pathlib import Path

parser = ArgumentParser(description="Generate CITATION.cff")
parser.add_argument("release_version", type=str)
release_version = parser.parse_args().release_version

out_dir = Path(__file__).parents[1]

# NOTE: ../CITATION.cff should not be continuously updated.
#       Run this script only at release time.

package_name = "MNE-Connectivity"
release_date = str(date.today())
commit = subprocess.run(
    ["git", "log", "-1", "--pretty=%H"], capture_output=True, text=True
).stdout.strip()

# KEYWORDS
keywords = (
    "MEG",
    "magnetoencephalography",
    "EEG",
    "electroencephalography",
    "fNIRS",
    "functional near-infrared spectroscopy",
    "iEEG",
    "intracranial EEG",
    "eCoG",
    "electrocorticography",
    "DBS",
    "deep brain stimulation",
    "connectivity",
    "functional connectivity",
    "effective connectivity",
    "coherence",
    "PLV",
    "phase-locking value",
    "PLI",
    "phase lag index",
    "PSI",
    "phase slope index",
    "Granger causality",
    "VAR",
    "vector autoregressive models",
)

# add to these as necessary
compound_surnames = ("van Vliet",)


def parse_name(name):
    """Split name blobs from `git shortlog -nse` into first/last/email."""
    # remove commit count
    _, name_and_email = name.strip().split("\t")
    name, email = name_and_email.split(" <")
    email = email.strip(">")
    email = "" if "noreply" in email else email  # ignore "noreply" emails
    name = " ".join(name.split("."))  # remove periods from initials
    # handle compound surnames
    for compound_surname in compound_surnames:
        if name.endswith(compound_surname):
            ix = name.index(compound_surname)
            first = name[:ix].strip()
            last = compound_surname
            return (first, last, email)
    # handle non-compound surnames
    name_elements = name.split()
    if len(name_elements) == 1:  # mononyms / usernames
        first = ""
        last = name
    else:
        first = " ".join(name_elements[:-1])
        last = name_elements[-1]
    return (first, last, email)


# MAKE SURE THE RELEASE STRING IS PROPERLY FORMATTED
try:
    split_version = list(map(int, release_version.split(".")))
except ValueError:
    raise
msg = (
    "First argument must be the release version X.Y.Z (all integers), "
    f"got {release_version}"
)
assert len(split_version) == 3, msg


# RUN GIT SHORTLOG TO GET ALL AUTHORS, SORTED BY NUMBER OF COMMITS
args = ["git", "shortlog", "-nse"]
result = subprocess.run(args, capture_output=True, text=True)
lines = result.stdout.strip().split("\n")
all_names = [parse_name(line) for line in lines if "[bot]" not in line]


# GENERATE CITATION.CFF
message = "If you use this software, please cite it using the following information."

# in CFF, multi-word keywords need to be wrapped in quotes
cff_keywords = (f'"{kw}"' if " " in kw else kw for kw in keywords)
# make into a bulleted list
cff_keywords = "\n".join(f"  - {kw}" for kw in cff_keywords)

# TODO: someday would be nice to include ORCiD identifiers too
cff_authors = [
    f"  - family-names: {last}\n    given-names: {first}"
    if first
    else f"  - name: {last}"
    for (first, last, _) in all_names
]
cff_authors = "\n".join(cff_authors)

# this ↓↓↓ is the meta-DOI that always resolves to the latest release
zenodo_doi = "10.5281/zenodo.10278399"

# ASSEMBLE THE CFF STRING
cff_boilerplate = f"""\
cff-version: 1.2.0
title: "{package_name}"
message: "{message}"
version: {release_version}
date-released: "{release_date}"
commit: {commit}
doi: {zenodo_doi}
keywords:
{cff_keywords}
authors:
{cff_authors}
"""

# WRITE TO FILE
with open(out_dir / "CITATION.cff", "w") as cff_file:
    cff_file.write(cff_boilerplate)

# UPDATE PACKAGE CITATION IN REFERENCES
bibtex_authors = []
for first, last, _ in all_names:
    if re.match(r".*\s.$", first):
        first += "."  # add period to initials
    bibtex_authors.append(first + ", " + last)
bibtex_authors = " and ".join(bibtex_authors)
bibtex_boilerplate = f"""\
@software{{MNE-Connectivity,
 author = {{{bibtex_authors}}},
 doi = {{{zenodo_doi}}},
 title = {{{{{package_name}}}}},
 year = {{{release_date[:4]}}}
}}
"""
references_path = out_dir / "doc" / "references.bib"
references = references_path.read_text()
references = re.sub(
    r"@software{MNE-Connectivity,.*?}\n\}\n",
    bibtex_boilerplate,
    references,
    flags=re.DOTALL,
)
references_path.write_text(references)
