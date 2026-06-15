# Velixar Memory API — Client Contract (authoritative)

The live API (`https://api.velixarai.com`, Azure Container Apps) **requires the `/v1`
prefix on every route.** All clients (Python/JS/Go SDKs + the MCP server) normalize
paths to `/v1` in their request layer — matching the server router in
`api/public_api.py` (which prepends `/v1` the same way). Derived from the server
router on 2026-06-14 after the SDK `/v1`-drift incident (clients were 404ing in prod).

| Operation        | Method | Path                                   |
|------------------|--------|----------------------------------------|
| store            | POST   | `/v1/memory`                           |
| get              | GET    | `/v1/memory/{id}`                      |
| update           | PATCH  | `/v1/memory/{id}`                      |
| delete           | DELETE | `/v1/memory/{id}`                      |
| list             | GET    | `/v1/memory/list`                      |
| search           | GET    | `/v1/memory/search`                    |
| identity         | GET    | `/v1/memory/identity`                  |
| graph_search     | POST   | `/v1/graph/search`                     |
| graph_traverse   | POST   | `/v1/graph/traverse` ⚠️ 404 live until the KG store is restored (Item A) |
| graph_entities   | GET    | `/v1/graph/entities`                   |
| graph_stats      | GET    | `/v1/graph/stats`                      |
| overview         | GET    | `/v1/exocortex/overview`               |
| health           | GET    | `/v1/health` (also reachable at `/health`) |

**Rule:** clients MUST prepend `/v1` idempotently in the request layer — never
hand-write `/v1` into each method. `tests/test_v1_contract.py` guards this.
