PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS collections (
  collection_id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  label TEXT,
  notes TEXT,
  source_priority_json TEXT NOT NULL DEFAULT '[]',
  raw_config_json TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ingest_runs (
  run_id INTEGER PRIMARY KEY AUTOINCREMENT,
  collection_id INTEGER NOT NULL,
  source TEXT NOT NULL,
  status TEXT NOT NULL,
  query_json TEXT,
  page_size INTEGER,
  limit_requested INTEGER,
  processed_hits INTEGER NOT NULL DEFAULT 0,
  works_created INTEGER NOT NULL DEFAULT 0,
  works_updated INTEGER NOT NULL DEFAULT 0,
  citations_written INTEGER NOT NULL DEFAULT 0,
  raw_dir TEXT,
  notes TEXT,
  started_at TEXT DEFAULT CURRENT_TIMESTAMP,
  finished_at TEXT,
  FOREIGN KEY (collection_id) REFERENCES collections(collection_id)
);

CREATE TABLE IF NOT EXISTS graph_build_runs (
  build_id INTEGER PRIMARY KEY AUTOINCREMENT,
  build_kind TEXT NOT NULL,
  status TEXT NOT NULL,
  notes TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  finished_at TEXT
);

CREATE TABLE IF NOT EXISTS dirty_objects (
  dirty_id INTEGER PRIMARY KEY AUTOINCREMENT,
  lane TEXT NOT NULL,
  object_kind TEXT NOT NULL,
  object_id INTEGER NOT NULL,
  collection_id INTEGER,
  reason TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(lane, object_kind, object_id),
  FOREIGN KEY (collection_id) REFERENCES collections(collection_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS maintenance_jobs (
  job_id INTEGER PRIMARY KEY AUTOINCREMENT,
  lane TEXT NOT NULL,
  status TEXT NOT NULL,
  scope TEXT,
  collection_name TEXT,
  updated_since TEXT,
  details_json TEXT,
  requested_at TEXT DEFAULT CURRENT_TIMESTAMP,
  started_at TEXT DEFAULT CURRENT_TIMESTAMP,
  finished_at TEXT,
  result_json TEXT
);

CREATE TABLE IF NOT EXISTS works (
  work_id INTEGER PRIMARY KEY AUTOINCREMENT,
  canonical_source TEXT NOT NULL,
  canonical_id TEXT NOT NULL,
  title TEXT NOT NULL,
  title_normalized TEXT,
  abstract TEXT,
  year INTEGER,
  citation_count INTEGER,
  primary_source_url TEXT,
  primary_pdf_url TEXT,
  raw_metadata_json TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(canonical_source, canonical_id)
);

CREATE TABLE IF NOT EXISTS work_ids (
  id_type TEXT NOT NULL,
  id_value TEXT NOT NULL,
  work_id INTEGER NOT NULL,
  is_primary INTEGER NOT NULL DEFAULT 0,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id_type, id_value),
  FOREIGN KEY (work_id) REFERENCES works(work_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS authors (
  author_id INTEGER PRIMARY KEY AUTOINCREMENT,
  display_name TEXT NOT NULL UNIQUE,
  raw_json TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS work_authors (
  work_id INTEGER NOT NULL,
  author_position INTEGER NOT NULL,
  author_id INTEGER NOT NULL,
  affiliations_json TEXT NOT NULL DEFAULT '[]',
  raw_json TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (work_id, author_position),
  FOREIGN KEY (work_id) REFERENCES works(work_id) ON DELETE CASCADE,
  FOREIGN KEY (author_id) REFERENCES authors(author_id)
);

CREATE TABLE IF NOT EXISTS collaborations (
  collaboration_id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS work_collaborations (
  work_id INTEGER NOT NULL,
  collaboration_id INTEGER NOT NULL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (work_id, collaboration_id),
  FOREIGN KEY (work_id) REFERENCES works(work_id) ON DELETE CASCADE,
  FOREIGN KEY (collaboration_id) REFERENCES collaborations(collaboration_id)
);

CREATE TABLE IF NOT EXISTS venues (
  venue_id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  venue_type TEXT,
  raw_json TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS work_venues (
  work_id INTEGER NOT NULL,
  venue_id INTEGER NOT NULL,
  is_primary INTEGER NOT NULL DEFAULT 1,
  raw_json TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (work_id, venue_id),
  FOREIGN KEY (work_id) REFERENCES works(work_id) ON DELETE CASCADE,
  FOREIGN KEY (venue_id) REFERENCES venues(venue_id)
);

CREATE TABLE IF NOT EXISTS topics (
  topic_id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT NOT NULL,
  topic_key TEXT NOT NULL,
  label TEXT NOT NULL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(source, topic_key)
);

CREATE TABLE IF NOT EXISTS work_topics (
  work_id INTEGER NOT NULL,
  topic_id INTEGER NOT NULL,
  score REAL,
  raw_json TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (work_id, topic_id),
  FOREIGN KEY (work_id) REFERENCES works(work_id) ON DELETE CASCADE,
  FOREIGN KEY (topic_id) REFERENCES topics(topic_id)
);

CREATE TABLE IF NOT EXISTS collection_works (
  collection_id INTEGER NOT NULL,
  work_id INTEGER NOT NULL,
  added_at TEXT DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (collection_id, work_id),
  FOREIGN KEY (collection_id) REFERENCES collections(collection_id) ON DELETE CASCADE,
  FOREIGN KEY (work_id) REFERENCES works(work_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS work_families (
  family_id INTEGER PRIMARY KEY AUTOINCREMENT,
  family_key TEXT NOT NULL UNIQUE,
  label TEXT,
  primary_work_id INTEGER,
  relation_kind TEXT NOT NULL DEFAULT 'standalone',
  confidence REAL,
  reason_json TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (primary_work_id) REFERENCES works(work_id)
);

CREATE TABLE IF NOT EXISTS work_family_members (
  family_id INTEGER NOT NULL,
  work_id INTEGER NOT NULL UNIQUE,
  member_role TEXT NOT NULL DEFAULT 'standalone',
  confidence REAL,
  reason_json TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (family_id, work_id),
  FOREIGN KEY (family_id) REFERENCES work_families(family_id) ON DELETE CASCADE,
  FOREIGN KEY (work_id) REFERENCES works(work_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS citations (
  citation_id INTEGER PRIMARY KEY AUTOINCREMENT,
  src_work_id INTEGER NOT NULL,
  dst_work_id INTEGER,
  dst_source TEXT,
  dst_external_id TEXT,
  raw_json TEXT,
  resolution_status TEXT NOT NULL DEFAULT 'unresolved',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (src_work_id) REFERENCES works(work_id) ON DELETE CASCADE,
  FOREIGN KEY (dst_work_id) REFERENCES works(work_id)
);

CREATE TABLE IF NOT EXISTS similarity_edges (
  src_work_id INTEGER NOT NULL,
  dst_work_id INTEGER NOT NULL,
  metric TEXT NOT NULL,
  score REAL NOT NULL,
  build_id INTEGER,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (src_work_id, dst_work_id, metric),
  FOREIGN KEY (src_work_id) REFERENCES works(work_id) ON DELETE CASCADE,
  FOREIGN KEY (dst_work_id) REFERENCES works(work_id) ON DELETE CASCADE,
  FOREIGN KEY (build_id) REFERENCES graph_build_runs(build_id)
);

CREATE TABLE IF NOT EXISTS work_embeddings (
  work_id INTEGER NOT NULL,
  embedding_model TEXT NOT NULL,
  vector_path TEXT,
  dim INTEGER,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (work_id, embedding_model),
  FOREIGN KEY (work_id) REFERENCES works(work_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS bibliographic_coupling_edges (
  src_work_id INTEGER NOT NULL,
  dst_work_id INTEGER NOT NULL,
  shared_reference_count INTEGER NOT NULL,
  score REAL,
  build_id INTEGER,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (src_work_id, dst_work_id),
  FOREIGN KEY (src_work_id) REFERENCES works(work_id) ON DELETE CASCADE,
  FOREIGN KEY (dst_work_id) REFERENCES works(work_id) ON DELETE CASCADE,
  FOREIGN KEY (build_id) REFERENCES graph_build_runs(build_id)
);

CREATE TABLE IF NOT EXISTS co_citation_edges (
  src_work_id INTEGER NOT NULL,
  dst_work_id INTEGER NOT NULL,
  shared_citer_count INTEGER NOT NULL,
  score REAL,
  build_id INTEGER,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (src_work_id, dst_work_id),
  FOREIGN KEY (src_work_id) REFERENCES works(work_id) ON DELETE CASCADE,
  FOREIGN KEY (dst_work_id) REFERENCES works(work_id) ON DELETE CASCADE,
  FOREIGN KEY (build_id) REFERENCES graph_build_runs(build_id)
);

CREATE TABLE IF NOT EXISTS documents (
  document_id INTEGER PRIMARY KEY AUTOINCREMENT,
  work_id INTEGER NOT NULL UNIQUE,
  parser_name TEXT,
  parser_version TEXT,
  parse_status TEXT NOT NULL DEFAULT 'registered',
  parsed_dir TEXT,
  manifest_path TEXT,
  parse_error TEXT,
  last_parse_attempt_at TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (work_id) REFERENCES works(work_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS document_sections (
  section_id INTEGER PRIMARY KEY AUTOINCREMENT,
  document_id INTEGER NOT NULL,
  parent_section_id INTEGER,
  ordinal TEXT,
  title TEXT,
  clean_title TEXT,
  path_text TEXT,
  section_kind TEXT NOT NULL DEFAULT 'body',
  level INTEGER NOT NULL DEFAULT 0,
  order_index INTEGER NOT NULL DEFAULT 0,
  page_start INTEGER,
  page_end INTEGER,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE,
  FOREIGN KEY (parent_section_id) REFERENCES document_sections(section_id)
);

CREATE TABLE IF NOT EXISTS blocks (
  block_id INTEGER PRIMARY KEY AUTOINCREMENT,
  document_id INTEGER NOT NULL,
  section_id INTEGER,
  block_type TEXT NOT NULL,
  page INTEGER,
  order_index INTEGER NOT NULL DEFAULT 0,
  text TEXT,
  raw_text TEXT,
  clean_text TEXT,
  text_level INTEGER,
  block_role TEXT NOT NULL DEFAULT 'body',
  is_heading INTEGER NOT NULL DEFAULT 0,
  is_retrievable INTEGER NOT NULL DEFAULT 1,
  exclusion_reason TEXT,
  latex TEXT,
  caption TEXT,
  asset_path TEXT,
  flags_json TEXT,
  raw_json TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE,
  FOREIGN KEY (section_id) REFERENCES document_sections(section_id)
);

CREATE TABLE IF NOT EXISTS formulas (
  formula_id INTEGER PRIMARY KEY AUTOINCREMENT,
  document_id INTEGER NOT NULL,
  section_id INTEGER,
  block_id INTEGER,
  page INTEGER,
  order_index INTEGER NOT NULL DEFAULT 0,
  latex TEXT NOT NULL,
  normalized_latex TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE,
  FOREIGN KEY (section_id) REFERENCES document_sections(section_id),
  FOREIGN KEY (block_id) REFERENCES blocks(block_id)
);

CREATE TABLE IF NOT EXISTS assets (
  asset_id INTEGER PRIMARY KEY AUTOINCREMENT,
  document_id INTEGER NOT NULL,
  section_id INTEGER,
  block_id INTEGER,
  asset_type TEXT NOT NULL,
  page INTEGER,
  caption TEXT,
  asset_path TEXT,
  raw_json TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE,
  FOREIGN KEY (section_id) REFERENCES document_sections(section_id),
  FOREIGN KEY (block_id) REFERENCES blocks(block_id)
);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
  work_id INTEGER NOT NULL,
  document_id INTEGER NOT NULL,
  section_id INTEGER,
  block_start_id INTEGER,
  block_end_id INTEGER,
  chunk_role TEXT NOT NULL DEFAULT 'section_child',
  page_hint TEXT,
  section_hint TEXT,
  text TEXT NOT NULL,
  raw_text TEXT,
  clean_text TEXT,
  text_hash TEXT,
  is_retrievable INTEGER NOT NULL DEFAULT 1,
  exclusion_reason TEXT,
  source_block_ids_json TEXT,
  flags_json TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (work_id) REFERENCES works(work_id) ON DELETE CASCADE,
  FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE,
  FOREIGN KEY (section_id) REFERENCES document_sections(section_id),
  FOREIGN KEY (block_start_id) REFERENCES blocks(block_id),
  FOREIGN KEY (block_end_id) REFERENCES blocks(block_id)
);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
  chunk_id INTEGER NOT NULL,
  embedding_model TEXT NOT NULL,
  vector_path TEXT,
  dim INTEGER,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (chunk_id, embedding_model),
  FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chunk_topic_mentions (
  chunk_id INTEGER NOT NULL,
  topic_id INTEGER NOT NULL,
  mention_count INTEGER NOT NULL DEFAULT 1,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (chunk_id, topic_id),
  FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE,
  FOREIGN KEY (topic_id) REFERENCES topics(topic_id)
);

CREATE TABLE IF NOT EXISTS work_capsules (
  capsule_id INTEGER PRIMARY KEY AUTOINCREMENT,
  work_id INTEGER NOT NULL UNIQUE,
  collection_id INTEGER,
  profile TEXT NOT NULL DEFAULT 'default',
  builder TEXT,
  is_review INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL,
  capsule_text TEXT NOT NULL,
  result_signature_json TEXT NOT NULL DEFAULT '[]',
  method_signature_json TEXT NOT NULL DEFAULT '[]',
  anomaly_code TEXT,
  anomaly_detail TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (work_id) REFERENCES works(work_id) ON DELETE CASCADE,
  FOREIGN KEY (collection_id) REFERENCES collections(collection_id) ON DELETE SET NULL
);


CREATE TABLE IF NOT EXISTS result_objects (
  result_object_id INTEGER PRIMARY KEY AUTOINCREMENT,
  work_id INTEGER NOT NULL,
  collection_id INTEGER,
  object_key TEXT,
  label TEXT NOT NULL,
  result_kind TEXT,
  summary_text TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'extracted',
  confidence REAL,
  signature_json TEXT NOT NULL DEFAULT '[]',
  evidence_json TEXT NOT NULL DEFAULT '[]',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (work_id) REFERENCES works(work_id) ON DELETE CASCADE,
  FOREIGN KEY (collection_id) REFERENCES collections(collection_id) ON DELETE SET NULL,
  UNIQUE(work_id, object_key)
);

CREATE TABLE IF NOT EXISTS result_values (
  result_value_id INTEGER PRIMARY KEY AUTOINCREMENT,
  result_object_id INTEGER NOT NULL,
  value_label TEXT NOT NULL,
  value_text TEXT,
  numeric_value REAL,
  unit_text TEXT,
  comparator TEXT,
  uncertainty_text TEXT,
  context_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (result_object_id) REFERENCES result_objects(result_object_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS result_context (
  result_context_id INTEGER PRIMARY KEY AUTOINCREMENT,
  result_object_id INTEGER NOT NULL,
  section_hint TEXT,
  dataset_hint TEXT,
  selection_hint TEXT,
  raw_context_json TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (result_object_id) REFERENCES result_objects(result_object_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS method_objects (
  method_object_id INTEGER PRIMARY KEY AUTOINCREMENT,
  work_id INTEGER NOT NULL,
  collection_id INTEGER,
  object_key TEXT,
  name TEXT NOT NULL,
  method_family TEXT,
  summary_text TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'extracted',
  signature_json TEXT NOT NULL DEFAULT '[]',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (work_id) REFERENCES works(work_id) ON DELETE CASCADE,
  FOREIGN KEY (collection_id) REFERENCES collections(collection_id) ON DELETE SET NULL,
  UNIQUE(work_id, object_key)
);

CREATE TABLE IF NOT EXISTS method_signatures (
  method_signature_id INTEGER PRIMARY KEY AUTOINCREMENT,
  method_object_id INTEGER NOT NULL,
  signature_kind TEXT NOT NULL,
  signature_text TEXT NOT NULL,
  normalized_text TEXT,
  raw_signature_json TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (method_object_id) REFERENCES method_objects(method_object_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS method_application_links (
  method_application_link_id INTEGER PRIMARY KEY AUTOINCREMENT,
  method_object_id INTEGER NOT NULL,
  result_object_id INTEGER,
  target_work_id INTEGER,
  relation_kind TEXT NOT NULL DEFAULT 'applied_to',
  confidence REAL,
  notes TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (method_object_id) REFERENCES method_objects(method_object_id) ON DELETE CASCADE,
  FOREIGN KEY (result_object_id) REFERENCES result_objects(result_object_id) ON DELETE SET NULL,
  FOREIGN KEY (target_work_id) REFERENCES works(work_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS transfer_candidates (
  transfer_candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_method_object_id INTEGER,
  source_result_object_id INTEGER,
  target_work_id INTEGER,
  target_context_json TEXT NOT NULL DEFAULT '{}',
  rationale_text TEXT,
  status TEXT NOT NULL DEFAULT 'proposed',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (source_method_object_id) REFERENCES method_objects(method_object_id) ON DELETE SET NULL,
  FOREIGN KEY (source_result_object_id) REFERENCES result_objects(result_object_id) ON DELETE SET NULL,
  FOREIGN KEY (target_work_id) REFERENCES works(work_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS transfer_edges (
  transfer_edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
  transfer_candidate_id INTEGER NOT NULL,
  src_method_object_id INTEGER,
  dst_work_id INTEGER,
  edge_kind TEXT NOT NULL DEFAULT 'candidate',
  score REAL,
  evidence_json TEXT NOT NULL DEFAULT '[]',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (transfer_candidate_id) REFERENCES transfer_candidates(transfer_candidate_id) ON DELETE CASCADE,
  FOREIGN KEY (src_method_object_id) REFERENCES method_objects(method_object_id) ON DELETE SET NULL,
  FOREIGN KEY (dst_work_id) REFERENCES works(work_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS reasoning_sessions (
  reasoning_session_id TEXT PRIMARY KEY,
  collection_id INTEGER,
  request_kind TEXT NOT NULL,
  request_json TEXT NOT NULL DEFAULT '{}',
  trace_mode TEXT NOT NULL DEFAULT 'structured_summary',
  raw_trace_enabled INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'created',
  started_at TEXT DEFAULT CURRENT_TIMESTAMP,
  finished_at TEXT,
  FOREIGN KEY (collection_id) REFERENCES collections(collection_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS reasoning_steps (
  reasoning_step_id INTEGER PRIMARY KEY AUTOINCREMENT,
  reasoning_session_id TEXT NOT NULL,
  step_index INTEGER NOT NULL,
  step_type TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'recorded',
  summary_text TEXT NOT NULL DEFAULT '',
  model_name TEXT,
  object_refs_json TEXT NOT NULL DEFAULT '[]',
  evidence_refs_json TEXT NOT NULL DEFAULT '[]',
  payload_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (reasoning_session_id) REFERENCES reasoning_sessions(reasoning_session_id) ON DELETE CASCADE,
  UNIQUE(reasoning_session_id, step_index)
);

CREATE TABLE IF NOT EXISTS reasoning_artifacts (
  reasoning_artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
  reasoning_session_id TEXT NOT NULL,
  reasoning_step_id INTEGER,
  artifact_kind TEXT NOT NULL,
  artifact_key TEXT,
  artifact_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (reasoning_session_id) REFERENCES reasoning_sessions(reasoning_session_id) ON DELETE CASCADE,
  FOREIGN KEY (reasoning_step_id) REFERENCES reasoning_steps(reasoning_step_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS idea_candidates (
  idea_candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
  reasoning_session_id TEXT,
  title TEXT NOT NULL,
  hypothesis_text TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'draft',
  rank_order INTEGER,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (reasoning_session_id) REFERENCES reasoning_sessions(reasoning_session_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS idea_scores (
  idea_score_id INTEGER PRIMARY KEY AUTOINCREMENT,
  idea_candidate_id INTEGER NOT NULL,
  score_axis TEXT NOT NULL,
  score_value REAL NOT NULL,
  scorer_kind TEXT NOT NULL DEFAULT 'heuristic',
  rationale_text TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (idea_candidate_id) REFERENCES idea_candidates(idea_candidate_id) ON DELETE CASCADE,
  UNIQUE(idea_candidate_id, score_axis, scorer_kind)
);

CREATE TABLE IF NOT EXISTS idea_evidence_links (
  idea_evidence_link_id INTEGER PRIMARY KEY AUTOINCREMENT,
  idea_candidate_id INTEGER NOT NULL,
  evidence_kind TEXT NOT NULL,
  evidence_id TEXT NOT NULL,
  supports INTEGER NOT NULL DEFAULT 1,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (idea_candidate_id) REFERENCES idea_candidates(idea_candidate_id) ON DELETE CASCADE,
  UNIQUE(idea_candidate_id, evidence_kind, evidence_id)
);

CREATE TABLE IF NOT EXISTS pdg_sources (
  source_id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  manifest_path TEXT,
  parsed_dir TEXT,
  block_count INTEGER NOT NULL DEFAULT 0,
  capsule_count INTEGER NOT NULL DEFAULT 0,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS pdg_sections (
  pdg_section_id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_id TEXT NOT NULL,
  parent_title TEXT,
  title TEXT NOT NULL,
  clean_title TEXT,
  path_text TEXT,
  section_kind TEXT NOT NULL DEFAULT 'body',
  level INTEGER NOT NULL DEFAULT 1,
  order_index INTEGER NOT NULL DEFAULT 0,
  page_start INTEGER,
  page_end INTEGER,
  raw_text TEXT,
  capsule_text TEXT NOT NULL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (source_id) REFERENCES pdg_sources(source_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS physics_concepts (
  physics_concept_id INTEGER PRIMARY KEY AUTOINCREMENT,
  concept_key TEXT NOT NULL UNIQUE,
  label TEXT NOT NULL,
  normalized_label TEXT NOT NULL,
  concept_kind TEXT NOT NULL DEFAULT 'section',
  source_kind TEXT NOT NULL DEFAULT 'pdg_seed',
  source_ref TEXT,
  summary_text TEXT NOT NULL DEFAULT '',
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS physics_aliases (
  physics_alias_id INTEGER PRIMARY KEY AUTOINCREMENT,
  physics_concept_id INTEGER NOT NULL,
  alias_text TEXT NOT NULL,
  normalized_alias TEXT NOT NULL,
  alias_kind TEXT NOT NULL DEFAULT 'label',
  confidence REAL NOT NULL DEFAULT 1.0,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(physics_concept_id, normalized_alias),
  FOREIGN KEY (physics_concept_id) REFERENCES physics_concepts(physics_concept_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS physics_relations (
  physics_relation_id INTEGER PRIMARY KEY AUTOINCREMENT,
  src_physics_concept_id INTEGER NOT NULL,
  dst_physics_concept_id INTEGER NOT NULL,
  relation_kind TEXT NOT NULL,
  weight REAL NOT NULL DEFAULT 1.0,
  source_ref TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(src_physics_concept_id, dst_physics_concept_id, relation_kind),
  FOREIGN KEY (src_physics_concept_id) REFERENCES physics_concepts(physics_concept_id) ON DELETE CASCADE,
  FOREIGN KEY (dst_physics_concept_id) REFERENCES physics_concepts(physics_concept_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS work_physics_groundings (
  work_physics_grounding_id INTEGER PRIMARY KEY AUTOINCREMENT,
  work_id INTEGER NOT NULL,
  physics_concept_id INTEGER NOT NULL,
  match_kind TEXT NOT NULL DEFAULT 'alias',
  confidence REAL NOT NULL DEFAULT 0.0,
  matched_alias TEXT,
  evidence_text TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(work_id, physics_concept_id),
  FOREIGN KEY (work_id) REFERENCES works(work_id) ON DELETE CASCADE,
  FOREIGN KEY (physics_concept_id) REFERENCES physics_concepts(physics_concept_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS result_physics_groundings (
  result_physics_grounding_id INTEGER PRIMARY KEY AUTOINCREMENT,
  result_object_id INTEGER NOT NULL,
  physics_concept_id INTEGER NOT NULL,
  match_kind TEXT NOT NULL DEFAULT 'alias',
  confidence REAL NOT NULL DEFAULT 0.0,
  matched_alias TEXT,
  evidence_text TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(result_object_id, physics_concept_id),
  FOREIGN KEY (result_object_id) REFERENCES result_objects(result_object_id) ON DELETE CASCADE,
  FOREIGN KEY (physics_concept_id) REFERENCES physics_concepts(physics_concept_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chunk_physics_groundings (
  chunk_physics_grounding_id INTEGER PRIMARY KEY AUTOINCREMENT,
  chunk_id INTEGER NOT NULL,
  physics_concept_id INTEGER NOT NULL,
  match_kind TEXT NOT NULL DEFAULT 'alias',
  confidence REAL NOT NULL DEFAULT 0.0,
  matched_alias TEXT,
  evidence_text TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(chunk_id, physics_concept_id),
  FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE,
  FOREIGN KEY (physics_concept_id) REFERENCES physics_concepts(physics_concept_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_collections_name ON collections(name);
CREATE INDEX IF NOT EXISTS idx_dirty_objects_lane ON dirty_objects(lane, object_kind, updated_at);
CREATE INDEX IF NOT EXISTS idx_dirty_objects_collection ON dirty_objects(collection_id, lane);
CREATE INDEX IF NOT EXISTS idx_maintenance_jobs_lane ON maintenance_jobs(lane, status, requested_at);
CREATE INDEX IF NOT EXISTS idx_works_year ON works(year);
CREATE INDEX IF NOT EXISTS idx_works_title ON works(title);
CREATE INDEX IF NOT EXISTS idx_works_title_norm ON works(title_normalized);
CREATE INDEX IF NOT EXISTS idx_work_ids_work ON work_ids(work_id);
CREATE INDEX IF NOT EXISTS idx_work_authors_author ON work_authors(author_id);
CREATE INDEX IF NOT EXISTS idx_work_collaborations_collaboration ON work_collaborations(collaboration_id);
CREATE INDEX IF NOT EXISTS idx_work_topics_topic ON work_topics(topic_id);
CREATE INDEX IF NOT EXISTS idx_collection_works_work ON collection_works(work_id);
CREATE INDEX IF NOT EXISTS idx_work_families_primary ON work_families(primary_work_id);
CREATE INDEX IF NOT EXISTS idx_work_family_members_family ON work_family_members(family_id);
CREATE INDEX IF NOT EXISTS idx_citations_src ON citations(src_work_id);
CREATE INDEX IF NOT EXISTS idx_citations_dst_external ON citations(dst_source, dst_external_id);
CREATE INDEX IF NOT EXISTS idx_citations_dst_work ON citations(dst_work_id);
CREATE INDEX IF NOT EXISTS idx_documents_work ON documents(work_id);
CREATE INDEX IF NOT EXISTS idx_sections_document ON document_sections(document_id);
CREATE INDEX IF NOT EXISTS idx_blocks_document ON blocks(document_id);
CREATE INDEX IF NOT EXISTS idx_formulas_document ON formulas(document_id);
CREATE INDEX IF NOT EXISTS idx_assets_document ON assets(document_id);
CREATE INDEX IF NOT EXISTS idx_work_capsules_collection ON work_capsules(collection_id, status);

CREATE INDEX IF NOT EXISTS idx_result_objects_work ON result_objects(work_id, status);
CREATE INDEX IF NOT EXISTS idx_method_objects_work ON method_objects(work_id, status);
CREATE INDEX IF NOT EXISTS idx_transfer_candidates_target ON transfer_candidates(target_work_id, status);
CREATE INDEX IF NOT EXISTS idx_reasoning_sessions_status ON reasoning_sessions(status, started_at);
CREATE INDEX IF NOT EXISTS idx_reasoning_steps_session ON reasoning_steps(reasoning_session_id, step_index);
CREATE INDEX IF NOT EXISTS idx_idea_candidates_session ON idea_candidates(reasoning_session_id, rank_order);
CREATE INDEX IF NOT EXISTS idx_pdg_sections_source ON pdg_sections(source_id, order_index);
CREATE INDEX IF NOT EXISTS idx_physics_concepts_kind ON physics_concepts(concept_kind, label);
CREATE INDEX IF NOT EXISTS idx_physics_aliases_norm ON physics_aliases(normalized_alias);
CREATE INDEX IF NOT EXISTS idx_physics_relations_src ON physics_relations(src_physics_concept_id, relation_kind);
CREATE INDEX IF NOT EXISTS idx_work_physics_groundings_work ON work_physics_groundings(work_id, confidence);
CREATE INDEX IF NOT EXISTS idx_work_physics_groundings_concept ON work_physics_groundings(physics_concept_id, confidence);
CREATE INDEX IF NOT EXISTS idx_result_physics_groundings_result ON result_physics_groundings(result_object_id, confidence);
CREATE INDEX IF NOT EXISTS idx_result_physics_groundings_concept ON result_physics_groundings(physics_concept_id, confidence);
CREATE INDEX IF NOT EXISTS idx_chunk_physics_groundings_chunk ON chunk_physics_groundings(chunk_id, confidence);
CREATE INDEX IF NOT EXISTS idx_chunk_physics_groundings_concept ON chunk_physics_groundings(physics_concept_id, confidence);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_work ON chunks(work_id);
