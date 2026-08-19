"""HTTP surface: uploads, ingestion, querying, auth, and the abuse cases."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from jjrag.api.app import create_app
from jjrag.config import Settings
from jjrag.security import egress

POLICY = (
    b"# Retention Policy\n\nCustomer records are retained for seven years from "
    b"the date of last activity. Exceptions require approval from "
    b"compliance@acme.example under procedure DR-14.\n"
)


@pytest.fixture
def client(settings: Settings) -> TestClient:
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client
    egress.uninstall()


@pytest.fixture
def authed_client(settings: Settings) -> TestClient:
    settings.security.api_token = "test-token"
    app = create_app(settings)
    with TestClient(app) as test_client:
        test_client.headers.update({"Authorization": "Bearer test-token"})
        yield test_client
    egress.uninstall()


def upload(client: TestClient, name: str, data: bytes):
    return client.post("/api/documents", files={"files": (name, data, "text/plain")})


class TestHealthAndCompliance:
    def test_health_reports_index_and_model_state(self, client: TestClient) -> None:
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert body["index_version"] is None
        assert body["embedding_backend"] == "hashing"

    def test_compliance_endpoint_attests_no_third_party_apis(
        self, client: TestClient
    ) -> None:
        body = client.get("/api/compliance").json()
        assert body["posture"]["third_party_model_apis_enabled"] is False
        assert body["posture"]["generation_provider"] == "ollama"
        assert "catalog" in body

    def test_security_headers_are_set(self, client: TestClient) -> None:
        headers = client.get("/api/health").headers
        assert "default-src 'self'" in headers["content-security-policy"]
        assert headers["x-content-type-options"] == "nosniff"
        assert headers["x-frame-options"] == "DENY"


class TestUpload:
    def test_accepts_a_supported_document(self, client: TestClient) -> None:
        body = upload(client, "policy.md", POLICY).json()
        assert len(body["accepted"]) == 1 and not body["rejected"]

    def test_rejects_executable_content(self, client: TestClient) -> None:
        body = upload(client, "payload.txt", b"MZ\x90\x00 binary").json()
        assert not body["accepted"] and body["rejected"]
        assert "msdownload" in body["rejected"][0]["reason"]

    def test_rejects_unsupported_extension(self, client: TestClient) -> None:
        body = upload(client, "run.sh", b"#!/bin/sh\necho hi\n").json()
        assert body["rejected"] and "allowlist" in body["rejected"][0]["reason"]

    def test_path_traversal_in_the_filename_is_neutralised(
        self, client: TestClient, settings: Settings
    ) -> None:
        upload(client, "../../etc/passwd.txt", b"root:x:0:0\n")
        # Nothing may be written outside the inbox.
        assert not (settings.paths.data_dir.parent / "etc").exists()
        written = list(settings.paths.inbox_dir.iterdir())
        assert all(".." not in p.name for p in written)

    def test_oversized_upload_is_refused(
        self, client: TestClient, settings: Settings
    ) -> None:
        settings.ingest.max_file_bytes = 128
        body = upload(client, "big.txt", b"x" * 5_000).json()
        assert body["rejected"]


class TestIngestAndQuery:
    def test_full_flow_upload_ingest_list_query(self, client: TestClient) -> None:
        assert upload(client, "policy.md", POLICY).json()["accepted"]

        run = client.post("/api/ingest", json={"force": False})
        assert run.status_code == 200
        payload = run.json()
        assert payload["status"] == "succeeded"
        assert payload["documents"] == 1 and payload["chunks"] >= 1
        assert payload["redactions"] == {"email": 1}
        assert [s["name"] for s in payload["stages"]][0] == "scan"

        documents = client.get("/api/documents").json()
        assert documents[0]["filename"] == "policy.md"

        # Health now reflects the published index.
        assert client.get("/api/health").json()["index_version"] == 1

    def test_ingest_with_nothing_to_do_reports_the_gate_failure(
        self, client: TestClient
    ) -> None:
        response = client.post("/api/ingest", json={})
        assert response.status_code == 422
        assert response.json()["status"] == "failed"
        assert response.json()["error"]

    def test_query_before_indexing_explains_what_to_do(
        self, client: TestClient
    ) -> None:
        response = client.post("/api/query", json={"question": "anything"})
        assert response.status_code == 409
        assert "Upload documents" in response.json()["detail"]

    def test_query_without_a_local_model_fails_clearly(
        self, client: TestClient
    ) -> None:
        upload(client, "policy.md", POLICY)
        client.post("/api/ingest", json={})
        response = client.post("/api/query", json={"question": "retention?"})
        assert response.status_code == 503
        assert "local model" in response.json()["detail"].lower()

    def test_run_history_and_detail_are_available(self, client: TestClient) -> None:
        upload(client, "policy.md", POLICY)
        run_id = client.post("/api/ingest", json={}).json()["run_id"]

        assert client.get("/api/runs").json()[0]["run_id"] == run_id
        detail = client.get(f"/api/runs/{run_id}").json()
        assert detail["run_id"] == run_id and detail["stages"]

    def test_unknown_run_is_404(self, client: TestClient) -> None:
        assert client.get("/api/runs/run_missing").status_code == 404

    def test_invalid_query_payload_is_rejected(self, client: TestClient) -> None:
        assert client.post("/api/query", json={"question": ""}).status_code == 422


class TestDeletion:
    def test_document_can_be_erased(self, client: TestClient) -> None:
        upload(client, "policy.md", POLICY)
        client.post("/api/ingest", json={})
        doc_id = client.get("/api/documents").json()[0]["doc_id"]

        response = client.delete(f"/api/documents/{doc_id}")
        assert response.status_code == 200 and response.json()["deleted"]
        assert client.get("/api/documents").json() == []

    def test_deleting_an_unknown_document_is_404(self, client: TestClient) -> None:
        assert client.delete("/api/documents/doc_missing").status_code == 404


class TestAuth:
    def test_writes_require_a_token_when_one_is_configured(
        self, authed_client: TestClient
    ) -> None:
        unauthenticated = TestClient(authed_client.app)
        assert unauthenticated.post(
            "/api/documents", files={"files": ("a.txt", b"hello", "text/plain")}
        ).status_code == 401
        assert unauthenticated.post("/api/ingest", json={}).status_code == 401

    def test_wrong_token_is_forbidden(self, authed_client: TestClient) -> None:
        client = TestClient(authed_client.app)
        client.headers.update({"Authorization": "Bearer wrong"})
        assert client.post("/api/ingest", json={}).status_code == 403

    def test_correct_token_is_accepted(self, authed_client: TestClient) -> None:
        assert upload(authed_client, "policy.md", POLICY).status_code == 200

    def test_reads_stay_open_when_anonymous_read_is_allowed(
        self, authed_client: TestClient
    ) -> None:
        client = TestClient(authed_client.app)
        assert client.get("/api/documents").status_code == 200

    def test_audit_endpoint_requires_the_token(
        self, authed_client: TestClient
    ) -> None:
        client = TestClient(authed_client.app)
        assert client.get("/api/audit").status_code == 401
        assert authed_client.get("/api/audit").status_code == 200


class TestRateLimiting:
    def test_upload_rate_limit_is_enforced(
        self, settings: Settings
    ) -> None:
        settings.security.upload_rate_limit_per_minute = 2
        app = create_app(settings)
        with TestClient(app) as client:
            statuses = [
                upload(client, f"f{i}.txt", b"some text content here").status_code
                for i in range(4)
            ]
        egress.uninstall()
        assert 429 in statuses


class TestStreaming:
    """The SSE endpoint must emit sources even when generation is unavailable."""

    def test_stream_reports_sources_then_the_model_error(
        self, client: TestClient
    ) -> None:
        upload(client, "policy.md", POLICY)
        client.post("/api/ingest", json={})

        with client.stream(
            "POST", "/api/query/stream", json={"question": "retention"}
        ) as response:
            body = "".join(response.iter_text())

        events = [line.split(": ", 1)[1] for line in body.splitlines()
                  if line.startswith("event: ")]
        assert events[0] == "sources"
        assert "error" in events  # no Ollama in the test environment


class TestAttestationAccuracy:
    """The compliance endpoint must describe the *running* configuration."""

    def test_reports_the_embedding_backend_actually_in_use(
        self, client: TestClient
    ) -> None:
        posture = client.get("/api/compliance").json()["posture"]
        assert posture["embedding_backend"] == "hashing"
        assert posture["embedding_model"].startswith("hashing-")

    def test_reports_the_ollama_embedding_model_when_that_backend_is_set(
        self, settings: Settings
    ) -> None:
        settings.embedding.backend = "ollama"
        settings.embedding.ollama_model = "nomic-embed-text"
        assert settings.describe_compliance()["embedding_model"] == "nomic-embed-text"
