# Publish this repository on GitHub

## GitHub website

1. Create a new empty repository, preferably named `stratus-regime-aware-preprocessing`.
2. Do not add a README, license, or `.gitignore` on GitHub; they are already included here.
3. Extract this archive locally.
4. Open a terminal in the extracted folder.
5. Run:

```powershell
.\scripts\publish_repository.ps1 -RepositoryUrl "https://github.com/USERNAME/stratus-regime-aware-preprocessing.git"
```

Alternatively, use GitHub Desktop: **File → Add local repository → Publish repository**.

## Before making it public

- confirm that MIT is the intended software license;
- update `CITATION.cff` after acceptance/publication;
- add the final GitHub URL and, if available, a Zenodo DOI;
- verify that no raw participant data or local paths were added.
