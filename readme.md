# DocThink

DocThink is an internal RAG-based assistant for answering questions using Confluence-based internal documentation. It parses Confluence pages, builds an embedding-based search index, and uses a local LLM to generate answers tailored to specific user roles.
DocThink is designed to streamline internal knowledge access by combining semantic search and role-based access control. It’s ideal for teams that need fast, accurate answers from internal documentation, with results scoped by both relevance and permissions.


## Features

- Role-based filtering of knowledge sources
- Semantic search using chunked documents
- Local LLM integration for generating answers (via Mistral)
- Handles incomplete context gracefully
- Simple REST API

---

## Project Structure

```
.
├── application/            # Core logic: configs, use cases, services
├── core/models/            # Pydantic models for data exchange
├── infrastructure/         # DB and LLM adapters
├── presentation/api/       # FastAPI router
├── scripts/                # Scripts for parsing and indexing docs
├── data/                   # Parsed docs, vector DB, keyword cache
├── main.py                 # FastAPI entry point
├── run.py                  # Development entry point
├── docker-compose.yml      # Local LLM container
├── README.md
```

---

## Pipeline

1. **Parse Confluence Pages**

   ```bash
   python scripts/parse_confluence_urls.py scripts/urls.txt
   ```

   Downloads and converts Confluence HTML pages to Markdown JSON format.

2. **Index Documents**

   ```bash
   python scripts/load_json_to_db.py
   ```

   Splits content into chunks, extracts keywords, stores embeddings in ChromaDB.

3. **Start LLM (optional)**

   ```bash
   python start_docker.sh
   ```

   Brings up LocalAI serving a Mistral model.

4. **Run the app**

   ```bash
   python run.py
   ```

---

## Search Logic Explained

When a query is received via the `/api/ask` endpoint, the following logic is applied:

1. **Keyword Search:** Extracts keywords using KeyBERT. Matches those against a keyword index built during document ingestion.

2. **Candidate Chunk Filtering:** Chunks that match keywords are prioritized. The system attempts to intersect keyword matches with semantic similarity results. If no keyword matches exist, a fallback to full semantic search is triggered.

3. **Distance Check:** Vector distances are checked against a configurable threshold (`DISTANCE_THRESHOLD`, e.g. 0.8) to ensure relevance.

4. **Role Filtering:** Only documents tagged with one of the user's available roles are considered (`admin`, `developer`, etc).

5. **Document Resolution:**

   - If only one document source is relevant: All chunks from that source are merged (up to the model's context limit) and passed to the LLM, which generates a tailored response using only that content. Titles are prioritized during chunk selection.
   - If multiple documents are relevant: A helpful summary is returned with links and preview snippets.
   - If no relevant chunks are found: A default message is returned.

---

## Example Query and Response

### Request (no documents found):

```json
{
  "question": "how to manage users in Confluence data center",
  "available_roles": ["developer"]
}
```

### Response:

```json
{
   "answer": "No relevant information found in the internal knowledge base.",
   "sources": [],
   "is_complete": false
}
```

### Request (multiple documents found):

```json
{
  "question": "how to manage users in Confluence data center",
  "available_roles": ["admin", "developer"]
}
```

### Response:

```json
{
  "answer": "Multiple relevant documents were found:\n\n\nhttps://confluence.atlassian.com/display/DOC/Configuring+Confluence+Security:\nConfiguring Confluence Security | Confluence Data Center 9.5 | Atlassian Documentation\n\n\nhttps://confluence.atlassian.com/display/DOC/Writing+User+Macros:\nWriting User Macros | Confluence Data Center 9.5 | Atlassian Documentation\n\n\nhttps://confluence.atlassian.com/display/DOC/Permissions+and+Restrictions:\nPermissions and restrictions | Confluence Data Center 9.5 | Atlassian Documentation\n\n\nPlease clarify your question or follow the links for more details.",
  "sources": [
    "https://confluence.atlassian.com/display/DOC/Configuring+Confluence+Security",
    "https://confluence.atlassian.com/display/DOC/Writing+User+Macros",
    "https://confluence.atlassian.com/display/DOC/Permissions+and+Restrictions"
  ],
  "is_complete": false
}
```

### Request (only one document found):

```json
{
  "question": "how to manage users in Confluence data center",
  "available_roles": ["jurist"]
}
```

### Response:

```json
{
   "answer": "\nTo manage users in Confluence Data Center, you can add users, assign permissions, and collect users into groups. See the following topics:\n\n* [Add and Invite Users](/doc/add-and-invite-users-138313.html)\n* [Delete or Disable Users](/doc/delete-or-disable-users-138318.html)\n* [Managing Site-Wide Permissions and Groups](/doc/managing-site-wide-permissions-and-groups-295306156.html)\n\nWhen you add a user or group to Confluence, it will be added to the external directory too, based on your configuration options. See [Configuring User Directories](/doc/configuring-user-directories-229838212.html).\n\nI couldn't find the information in the available context.",
   "sources": [
      "https://confluence.atlassian.com/display/DOC/Manage+Users"
   ],
   "is_complete": true
}
```


