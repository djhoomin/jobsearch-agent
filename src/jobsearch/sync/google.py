"""Optional Gmail / Drive / Sheets sync.

Feature-flagged and entirely optional: **every other part of this tool works
with zero Google credentials configured**. The Google client libraries are an
extra (``pip install -e '.[google]'``) and are imported lazily inside functions,
so importing this module never fails on a machine that has not installed them.

Safety rules that are structural, not conventions:

* Gmail scope is ``gmail.compose``, which can create drafts but **cannot send**.
  There is no ``users().messages().send()`` call anywhere in this file.
* Drive scope is ``drive.file``, which only grants access to files this app
  itself created. It cannot read the user's existing Drive.
* Tokens and client secrets are written to paths that ``.gitignore`` excludes.

See the README's "Google setup" section for the Cloud Console walkthrough.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Sequence

log = logging.getLogger(__name__)

#: Least-privilege scopes. gmail.compose deliberately excludes gmail.send.
SCOPES = [
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/spreadsheets",
]

#: Read-only scope, added only when [google].gmail_read_replies is true.
GMAIL_READONLY = "https://www.googleapis.com/auth/gmail.readonly"


class GoogleSyncError(RuntimeError):
    """Google sync is unavailable, unconfigured, or failed."""


class GoogleSyncDisabled(GoogleSyncError):
    """Sync was requested but is switched off in config."""


@dataclass
class GoogleSync:
    """Thin wrapper over the three Google APIs this tool uses."""

    client_secret_path: Path
    token_path: Path
    drive_folder_name: str = "Tailored CVs"
    sheets_spreadsheet_id: str = ""
    read_replies: bool = False
    dry_run: bool = False
    _creds: Any = None

    @classmethod
    def from_config(cls, cfg, dry_run: bool = False) -> "GoogleSync":
        section = cfg.section("google")
        if not section.get("enabled", False):
            raise GoogleSyncDisabled(
                "Google sync is disabled. Set [google].enabled = true in config.toml "
                "after completing the Cloud Console setup in the README."
            )
        root = cfg.root
        return cls(
            client_secret_path=_resolve(root, section.get("client_secret_path", "client_secret.json")),
            token_path=_resolve(root, section.get("token_path", ".secrets/google_token.json")),
            drive_folder_name=section.get("drive_folder_name", "Tailored CVs"),
            sheets_spreadsheet_id=section.get("sheets_spreadsheet_id", ""),
            read_replies=bool(section.get("gmail_read_replies", False)),
            dry_run=dry_run,
        )

    # -- auth --------------------------------------------------------------
    def credentials(self) -> Any:
        """Installed-app OAuth flow, cached to ``token_path``."""
        if self._creds is not None:
            return self._creds
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
        except ImportError as exc:  # pragma: no cover - optional extra
            raise GoogleSyncError(
                "Google client libraries are not installed. Run:\n"
                "  pip install -e '.[google]'"
            ) from exc

        scopes = [*SCOPES, GMAIL_READONLY] if self.read_replies else list(SCOPES)
        creds = None
        if self.token_path.is_file():
            creds = Credentials.from_authorized_user_file(str(self.token_path), scopes)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        if not creds or not creds.valid:
            if not self.client_secret_path.is_file():
                raise GoogleSyncError(
                    f"No OAuth client secret at {self.client_secret_path}. "
                    "Follow the README's Google setup walkthrough."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(self.client_secret_path), scopes
            )
            creds = flow.run_local_server(port=0)
            self.token_path.parent.mkdir(parents=True, exist_ok=True)
            self.token_path.write_text(creds.to_json(), encoding="utf-8")
            self.token_path.chmod(0o600)
        self._creds = creds
        return creds

    def _service(self, name: str, version: str) -> Any:
        try:
            from googleapiclient.discovery import build
        except ImportError as exc:  # pragma: no cover - optional extra
            raise GoogleSyncError("pip install -e '.[google]' to enable Google sync") from exc
        return build(name, version, credentials=self.credentials(), cache_discovery=False)

    # -- gmail: DRAFTS ONLY -------------------------------------------------
    def create_draft(
        self, to: str, subject: str, body: str, attachments: Sequence[Path] = ()
    ) -> str:
        """Create a Gmail draft. This tool has no code path that sends mail."""
        if self.dry_run:
            log.info("[dry-run] would create a Gmail draft to %s: %s", to, subject)
            return "dry-run"

        message = EmailMessage()
        message["To"] = to
        message["Subject"] = subject
        message.set_content(body)
        for path in attachments:
            path = Path(path)
            if not path.is_file():
                log.warning("attachment missing, skipping: %s", path)
                continue
            message.add_attachment(
                path.read_bytes(),
                maintype="application",
                subtype="pdf" if path.suffix.lower() == ".pdf" else "octet-stream",
                filename=path.name,
            )
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        service = self._service("gmail", "v1")
        draft = (
            service.users()
            .drafts()
            .create(userId="me", body={"message": {"raw": raw}})
            .execute()
        )
        return draft.get("id", "")

    def find_replies(self, query: str = "newer_than:30d") -> list[dict[str, Any]]:
        """Read recent threads to spot application replies. Requires opt-in."""
        if not self.read_replies:
            raise GoogleSyncError(
                "Reading Gmail is off. Set [google].gmail_read_replies = true and "
                "re-authorise (delete the token file) to enable it."
            )
        if self.dry_run:
            log.info("[dry-run] would search Gmail for: %s", query)
            return []
        service = self._service("gmail", "v1")
        result = (
            service.users().messages().list(userId="me", q=query, maxResults=50).execute()
        )
        out: list[dict[str, Any]] = []
        for ref in result.get("messages", []):
            message = (
                service.users()
                .messages()
                .get(userId="me", id=ref["id"], format="metadata",
                     metadataHeaders=["From", "Subject", "Date"])
                .execute()
            )
            headers = {
                h["name"]: h["value"] for h in message.get("payload", {}).get("headers", [])
            }
            out.append(
                {
                    "id": ref["id"],
                    "from": headers.get("From", ""),
                    "subject": headers.get("Subject", ""),
                    "date": headers.get("Date", ""),
                    "snippet": message.get("snippet", ""),
                }
            )
        return out

    # -- drive --------------------------------------------------------------
    def ensure_folder(self) -> str:
        """Find or create the CV folder. drive.file scope keeps this app-scoped."""
        if self.dry_run:
            return "dry-run-folder"
        service = self._service("drive", "v3")
        query = (
            "mimeType='application/vnd.google-apps.folder' and trashed=false and "
            f"name='{self.drive_folder_name}'"
        )
        found = service.files().list(q=query, fields="files(id,name)", spaces="drive").execute()
        files = found.get("files", [])
        if files:
            return files[0]["id"]
        created = (
            service.files()
            .create(
                body={
                    "name": self.drive_folder_name,
                    "mimeType": "application/vnd.google-apps.folder",
                },
                fields="id",
            )
            .execute()
        )
        return created["id"]

    def upload_cv(self, pdf_path: str | Path) -> str:
        """Upload a tailored CV PDF. Returns its Drive web link."""
        path = Path(pdf_path)
        if not path.is_file():
            raise GoogleSyncError(f"No such file to upload: {path}")
        if self.dry_run:
            log.info("[dry-run] would upload %s to Drive folder %r", path, self.drive_folder_name)
            return "dry-run"
        from googleapiclient.http import MediaFileUpload

        service = self._service("drive", "v3")
        media = MediaFileUpload(str(path), mimetype="application/pdf", resumable=False)
        created = (
            service.files()
            .create(
                body={"name": path.name, "parents": [self.ensure_folder()]},
                media_body=media,
                fields="id,webViewLink",
            )
            .execute()
        )
        return created.get("webViewLink", created.get("id", ""))

    # -- sheets -------------------------------------------------------------
    def mirror_to_sheet(self, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
        """Replace the first sheet's contents with the tracker's Pipeline view."""
        if not self.sheets_spreadsheet_id:
            raise GoogleSyncError(
                "No [google].sheets_spreadsheet_id configured. Create a spreadsheet, "
                "copy its id out of the URL, and set it in config.toml."
            )
        if self.dry_run:
            log.info("[dry-run] would mirror %d rows to Sheets", len(rows))
            return "dry-run"
        service = self._service("sheets", "v4")
        values = [list(header), *[list(r) for r in rows]]
        service.spreadsheets().values().clear(
            spreadsheetId=self.sheets_spreadsheet_id, range="A:Z"
        ).execute()
        service.spreadsheets().values().update(
            spreadsheetId=self.sheets_spreadsheet_id,
            range="A1",
            valueInputOption="RAW",
            body={"values": values},
        ).execute()
        return f"https://docs.google.com/spreadsheets/d/{self.sheets_spreadsheet_id}"


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path)


__all__ = ["GoogleSync", "GoogleSyncDisabled", "GoogleSyncError", "SCOPES"]
