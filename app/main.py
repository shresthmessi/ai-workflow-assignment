from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 👇 IMPORTANT: use relative imports (the dot . means "from the same package")
from .engine.graph_engine import GraphEngine, ToolRegistry, RunStatus
from .workflows.code_review import register_code_review_tools


# ------------------------------------------------
# Create registry + register tools from workflow
# ------------------------------------------------

tool_registry = ToolRegistry()
register_code_review_tools(tool_registry)

engine = GraphEngine(tool_registry)


# ------------------------------------------------
# Pydantic Models for API Requests / Responses
# ------------------------------------------------

class NodeConfig(BaseModel):
    name: str = Field(..., description="Unique node name")
    tool: str = Field(..., description="Tool name to execute")


class CreateGraphRequest(BaseModel):
    nodes: List[NodeConfig]
    edges: Dict[str, Optional[str]] = Field(
        ..., description="Mapping: from_node -> to_node (or null)"
    )
    start_node: str = Field(..., description="The first node to run")


class CreateGraphResponse(BaseModel):
    graph_id: str


class RunGraphRequest(BaseModel):
    graph_id: str
    initial_state: Dict[str, Any] = Field(default_factory=dict)


class LogEntry(BaseModel):
    node: str
    tool: str
    state: Dict[str, Any]


class RunGraphResponse(BaseModel):
    run_id: str
    graph_id: str
    final_state: Dict[str, Any]
    log: List[LogEntry]
    status: RunStatus
    error: Optional[str]


class GetRunStateResponse(BaseModel):
    run_id: str
    graph_id: str
    current_node: Optional[str]
    state: Dict[str, Any]
    log: List[LogEntry]
    status: RunStatus
    error: Optional[str]


# ------------------------------------------------
# FastAPI Initialization
# ------------------------------------------------

app = FastAPI(
    title="Workflow Engine Assignment",
    description="Minimal agent workflow executor using FastAPI.",
    version="1.0",
)


# ------------------------------------------------
# Create Graph Endpoint
# ------------------------------------------------

@app.post("/graph/create", response_model=CreateGraphResponse)
def create_graph(req: CreateGraphRequest):

    # Build nodes dict: node_name -> tool_name
    nodes_map: Dict[str, str] = {}
    for node in req.nodes:
        if node.name in nodes_map:
            raise HTTPException(status_code=400, detail=f"Duplicate node name: {node.name}")
        nodes_map[node.name] = node.tool

    # Validate edges
    for from_node, to_node in req.edges.items():
        if from_node not in nodes_map:
            raise HTTPException(status_code=400, detail=f"Unknown node in edges: {from_node}")
        if to_node is not None and to_node not in nodes_map:
            raise HTTPException(status_code=400, detail=f"Unknown next node: {to_node}")

    if req.start_node not in nodes_map:
        raise HTTPException(status_code=400, detail="start_node must be one of the nodes")

    graph_id = engine.create_graph(
        nodes=nodes_map,
        edges=req.edges,
        start_node=req.start_node,
    )

    return CreateGraphResponse(graph_id=graph_id)


# ------------------------------------------------
# Run Graph Endpoint
# ------------------------------------------------

@app.post("/graph/run", response_model=RunGraphResponse)
def run_graph(req: RunGraphRequest):
    try:
        run = engine.start_run(req.graph_id, req.initial_state)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    log_entries = [
        LogEntry(node=entry["node"], tool=entry["tool"], state=entry["state"])
        for entry in run.log
    ]

    return RunGraphResponse(
        run_id=run.run_id,
        graph_id=run.graph.graph_id,
        final_state=run.state,
        log=log_entries,
        status=run.status,
        error=run.error,
    )


# ------------------------------------------------
# Get Run State Endpoint
# ------------------------------------------------

@app.get("/graph/state/{run_id}", response_model=GetRunStateResponse)
def get_run_state(run_id: str):
    try:
        run = engine.get_run(run_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    log_entries = [
        LogEntry(node=entry["node"], tool=entry["tool"], state=entry["state"])
        for entry in run.log
    ]

    return GetRunStateResponse(
        run_id=run.run_id,
        graph_id=run.graph.graph_id,
        current_node=run.current_node,
        state=run.state,
        log=log_entries,
        status=run.status,
        error=run.error,
    )


# ------------------------------------------------
# Helper endpoint to list available tools
# ------------------------------------------------

@app.get("/tools")
def list_tools():
    return {"tools": tool_registry.list_tools()}
