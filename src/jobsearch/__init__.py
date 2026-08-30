"""jobsearch-agent: an agent harness for a personal job search.

Pipeline stages, each independently invocable and chainable:

``discover``  public ATS job boards (Greenhouse / Lever / Ashby), never scrapers
``score``     hard constraints first, then the weighted rubric
``tailor``    role-tailored CV from the base CV + career dossier, grounded
``verify``    ATS verification of the rendered PDF
``outreach``  inferred contacts and drafted messages, never sent automatically
``track``     SQLite source of truth, exported to the user's xlsx shape
``sync``      optional Gmail drafts / Drive / Sheets
``run``       all of the above, agentically, via the Claude Tool Runner
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
