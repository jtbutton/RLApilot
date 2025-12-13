from typing import List, Optional
from pydantic import BaseModel


class ResourceIngestionReport(BaseModel):
    location: str
    resource_id: Optional[str]
    status: str
    validation_warnings: List[str] = []
    validation_errors: List[str] = []
    neo4j_summary: Optional[dict] = None
    postgres_summary: Optional[dict] = None
    vector_summary: Optional[dict] = None

    @classmethod
    def from_components(cls, location, resource_meta, validation_report, write_results):
        status = "success"
        if validation_report.errors:
            status = "partial"

        return cls(
            location=location,
            resource_id=resource_meta.id,
            status=status,
            validation_warnings=validation_report.warnings,
            validation_errors=validation_report.errors,
            neo4j_summary=write_results.get("neo4j"),
            postgres_summary=write_results.get("postgres"),
            vector_summary=write_results.get("vector"),
        )


class IngestionRunReport(BaseModel):
    run_id: Optional[str]
    resource_reports: List[ResourceIngestionReport] = []
    fatal_errors: List[dict] = []
    status: str = "running"

    @classmethod
    def start_new(cls, run_id: Optional[str] = None) -> "IngestionRunReport":
        return cls(run_id=run_id)

    def add_fatal_error(self, location: str, error: str) -> None:
        self.fatal_errors.append({"location": location, "error": error})

    def finalize(self) -> None:
        if self.fatal_errors:
            self.status = "completed_with_errors"
        else:
            self.status = "completed"
