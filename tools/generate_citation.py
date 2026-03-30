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
)

# add to these as necessary
compound_surnames = ("van Vliet",)

# DUPLICATE AUTHORS TO COMBINE INFO FOR
duplicate_authors = (
    dict(
        main=("Thomas S", "Binns", "t.s.binns@outlook.com"),
        copies=[("Thomas Samuel", "Binns", "t.s.binns@outlook.com")],
    ),
    dict(
        main=("Adam", "Li", "adam2392@gmail.com"),
        copies=[
            ("Adam", "Li", "adam2392@Adams-MacBook-Pro-2.local"),
            ("Adam", "Li", "adam2392@adams-mbp-2.lan"),
            ("Adam", "Li", "adam2392@new-host-2.home"),
            ("Adam", "Li", "adam2392@Adams-MBP-2.home"),
        ],
    ),
    dict(
        main=("Alex", "Rockhill", "aprockhill@mailbox.org"),
        copies=[("", "Alex", "aprockhill@mailbox.org")],
    ),
)

# AUTHORS TO FIX MISSING/MANGLED INFO FOR
fix_authors = (
    [("", "Mohammad", ""), ("Mohammad", "Orabe", "")],
    [("", "SezanMert", ""), ("Sezan", "Mert", "")],
)


def parse_name(name):
    """Split name blobs from `git shortlog -nse` into n_commits/first/last/email."""
    # remove commit count
    n_commits, name_and_email = name.strip().split("\t")
    n_commits = int(n_commits)
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
            return (first, last, email), n_commits
    # handle non-compound surnames
    name_elements = name.split()
    if len(name_elements) == 1:  # mononyms / usernames
        first = ""
        last = name
    else:
        first = " ".join(name_elements[:-1])
        last = name_elements[-1]
    return (first, last, email), n_commits


def combine_duplicates(names, n_commits, duplicate_authors):
    """Combine duplicate authors into a single author and re-sort by commits."""
    new_names = []
    new_n_commits = []
    for entry in duplicate_authors:
        main = entry["main"]
        copies = entry["copies"]
        main_name_idx = names.index(main)
        tot_n_commits = n_commits[main_name_idx]
        drop_idcs = [main_name_idx]
        for copy in copies:
            copy_name_idx = names.index(copy)
            tot_n_commits += n_commits[copy_name_idx]
            drop_idcs.append(copy_name_idx)
        # drop the duplicate (and original) entries
        for idx in sorted(drop_idcs, reverse=True):
            names.pop(idx)
            n_commits.pop(idx)
        new_names.append(main)
        new_n_commits.append(tot_n_commits)
    new_names.extend(names)
    new_n_commits.extend(n_commits)

    return new_names, new_n_commits


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
all_names = []
all_n_commits = []
for line in lines:
    if "[bot]" not in line:
        name, n_commits = parse_name(line)
        all_names.append(name)
        all_n_commits.append(n_commits)
all_names, all_n_commits = combine_duplicates(
    all_names, all_n_commits, duplicate_authors
)
all_names = sorted(
    all_names, key=lambda x: all_n_commits[all_names.index(x)], reverse=True
)
for old, new in fix_authors:
    idx = all_names.index(old)
    all_names[idx] = new


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
