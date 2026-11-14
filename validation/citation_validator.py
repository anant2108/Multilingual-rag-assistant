
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CitationReport:
    """Stores the results of citation integrity checks."""

    cited_ids: list[int]
    invalid_ids: list[int]
    source_map: list[dict]

    @property
    def references_are_valid(self):
        return len(self.invalid_ids) == 0


def validate_citations(answer: str, documents) -> CitationReport:
    """
    Check whether citation numbers refer to retrieved documents.

    This checks citation integrity only.
    It does not verify whether the cited passage supports the claim.
    """

    if not isinstance(answer, str):
        raise TypeError("Answer must be a string.")

    # Extract citation numbers such as [1], [2], [3].
    matches = re.findall(r"\[(\d+)\]", answer)

    cited_ids = sorted({
        int(value)
        for value in matches
    })

    # Citation numbers must correspond to retrieved documents.
    valid_ids = set(range(1, len(documents) + 1))

    invalid_ids = sorted(
        set(cited_ids) - valid_ids
    )

    # Build a map between citation numbers and their sources.
    source_map = []

    for citation_id in cited_ids:
        if citation_id not in valid_ids:
            continue

        document = documents[citation_id - 1]
        metadata = document.metadata or {}

        source = Path(
            str(metadata.get("source", "Unknown source"))
        ).name

        page = metadata.get("page")

        if isinstance(page, int):
            page = page + 1

        source_map.append({
            "citation_id": citation_id,
            "source": source,
            "page": page,
        })

    return CitationReport(
        cited_ids=cited_ids,
        invalid_ids=invalid_ids,
        source_map=source_map,
    )
