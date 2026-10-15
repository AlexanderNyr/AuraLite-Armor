# Contributing

Thanks for your interest in Armor Penetration Lab. The project is experimental; please distinguish implementation bugs from questions about physical validity.

## Before submitting a change

1. Keep source, comments, user-interface text, and documentation in English.
2. Add or update tests for behavior changes.
3. Install the test dependencies and run the Python unit tests from the repository root:

   ```bash
   python -m pip install -r requirements-test.txt
   python -m unittest discover -s tests -v
   ```

4. If you changed the C++ solver, build both dimensions and run the small smoke cases described in `.github/workflows/ci.yml`.
5. Describe what was tested. A successful build or smoke test is not evidence that a material model is physically validated.

## Issues and pull requests

Include your operating system, compiler/Python versions, exact command line, relevant `meta.txt` fields, and a minimal run directory or reproducible example when possible. Remove private or sensitive data before attaching files.
