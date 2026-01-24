
from fastapi.testclient import TestClient
from sqlmodel import Session

def test_create_route(client: TestClient, tenant_header):
    response = client.post(
        "/api/v1/tgflow/routes",
        headers=tenant_header,
        json={
            "route_key": "qa-route-alpha",
            "base_model_ref": "microsoft/phi-2",
            "description": "QA Automation Route"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["route_key"] == "qa-route-alpha"
    # assert data["enabled"] is True # Endpoint returns confirmation msg, not full object

def test_list_routes(client: TestClient, tenant_header):
    # Setup
    client.post("/api/v1/tgflow/routes", headers=tenant_header, json={"route_key": "r1", "base_model_ref": "m1"})
    client.post("/api/v1/tgflow/routes", headers=tenant_header, json={"route_key": "r2", "base_model_ref": "m1"})
    
    response = client.get("/api/v1/tgflow/routes", headers=tenant_header)
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 2
    keys = [r["route_key"] for r in data]
    assert "r1" in keys
    assert "r2" in keys

def test_create_duplicate_route_fails(client: TestClient, tenant_header):
    payload = {"route_key": "dup-route", "base_model_ref": "m1"}
    client.post("/api/v1/tgflow/routes", headers=tenant_header, json=payload)
    
    response = client.post("/api/v1/tgflow/routes", headers=tenant_header, json=payload)
    assert response.status_code == 400
    assert "already exists" in response.json().get("detail", "").lower()

def test_multi_tenant_isolation(client: TestClient):
    t1 = {"X-Tenant-ID": "t1"}
    t2 = {"X-Tenant-ID": "t2"}
    
    # Create in T1
    client.post("/api/v1/tgflow/routes", headers=t1, json={"route_key": "secret-route", "base_model_ref": "m1"})
    
    # List in T2
    response = client.get("/api/v1/tgflow/routes", headers=t2)
    data = response.json()
    # Should be empty or not contain secret-route
    assert not any(r["route_key"] == "secret-route" for r in data)
