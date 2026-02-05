# AG-UI Protocol - Agent-User Interaction Layer for Generative UI
# This module implements a bidirectional protocol for real-time UI generation

from enum import Enum
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
import json
import uuid


class UIComponentType(str, Enum):
    """Types of UI components agents can generate"""
    # Layout
    CARD = "card"
    CONTAINER = "container"
    GRID = "grid"
    COLUMN = "column"
    ROW = "row"
    DIVIDER = "divider"
    SPACER = "spacer"
    
    # Data Display
    TEXT = "text"
    CODE_BLOCK = "code_block"
    CODE_DIFF = "code_diff"
    TABLE = "table"
    LIST = "list"
    BADGE = "badge"
    STAT = "stat"
    PROGRESS = "progress"
    CHART = "chart"
    
    # Feedback
    ALERT = "alert"
    TOAST = "toast"
    TIMELINE = "timeline"
    STEP_INDICATOR = "step_indicator"
    
    # Interactive
    BUTTON = "button"
    BUTTON_GROUP = "button_group"
    FORM = "form"
    INPUT = "input"
    SELECT = "select"
    CHECKBOX = "checkbox"
    TOGGLE = "toggle"
    TABS = "tabs"
    ACCORDION = "accordion"
    EXPANDABLE = "expandable"
    
    # Agent-specific
    AGENT_STATUS = "agent_status"
    FIX_CARD = "fix_card"
    VULNERABILITY_CARD = "vulnerability_card"
    TEST_RESULT = "test_result"
    SECURITY_SCAN = "security_scan"
    WORKFLOW_GRAPH = "workflow_graph"


class UIEventType(str, Enum):
    """Events that can be sent between agent and UI"""
    # Lifecycle
    START = "start"
    PROGRESS = "progress"
    COMPLETE = "complete"
    ERROR = "error"
    
    # UI Updates
    RENDER = "render"
    UPDATE = "update"
    REMOVE = "remove"
    REPLACE = "replace"
    
    # User Actions
    CLICK = "click"
    SUBMIT = "submit"
    SELECT = "select"
    DISMISS = "dismiss"
    EXPAND = "expand"
    COLLAPSE = "collapse"
    
    # Data
    DATA_UPDATE = "data_update"
    STREAM_CHUNK = "stream_chunk"
    STREAM_END = "stream_end"


class Severity(str, Enum):
    """Severity levels for issues/fixes"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AgentPhase(str, Enum):
    """Phases of agent execution"""
    IDLE = "idle"
    STARTING = "starting"
    ANALYZING = "analyzing"
    PROCESSING = "processing"
    FIXING = "fixing"
    COMPLETE = "complete"
    ERROR = "error"


# ==================== UI Component Specs ====================

class UIStyle(BaseModel):
    """Styling options for UI components"""
    variant: Optional[str] = None  # primary, secondary, success, warning, danger, info
    size: Optional[str] = None  # sm, md, lg, xl
    color: Optional[str] = None
    background: Optional[str] = None
    border: Optional[str] = None
    padding: Optional[str] = None
    margin: Optional[str] = None
    animation: Optional[str] = None  # fadeIn, slideIn, pulse, none
    className: Optional[str] = None


class UIAction(BaseModel):
    """Action that can be triggered from UI"""
    name: str
    label: Optional[str] = None
    icon: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    confirm: Optional[str] = None  # Confirmation message


class UIComponent(BaseModel):
    """Base UI component specification"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: UIComponentType
    props: Dict[str, Any] = Field(default_factory=dict)
    children: Optional[List["UIComponent"]] = None
    style: Optional[UIStyle] = None
    actions: Optional[List[UIAction]] = None
    visible: bool = True
    
    class Config:
        use_enum_values = True


# ==================== Specialized Component Specs ====================

class CodeDiffSpec(BaseModel):
    """Specification for code diff component"""
    before: str
    after: str
    language: str = "python"
    title: Optional[str] = None
    lineNumbers: bool = True
    highlightChanges: bool = True


class FixCardSpec(BaseModel):
    """Specification for a fix card (shown when agent fixes code)"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent: str
    description: str
    severity: Severity
    category: Optional[str] = None
    line: Optional[int] = None
    before: Optional[str] = None
    after: Optional[str] = None
    explanation: Optional[str] = None
    applied: bool = True
    
    class Config:
        use_enum_values = True


class AgentStatusSpec(BaseModel):
    """Specification for agent status display"""
    agentId: str
    agentName: str
    icon: str
    phase: AgentPhase
    message: str
    progress: Optional[int] = None  # 0-100
    stats: Optional[Dict[str, Any]] = None
    fixes: Optional[List[FixCardSpec]] = None
    duration: Optional[float] = None  # seconds
    
    class Config:
        use_enum_values = True


class WorkflowStepSpec(BaseModel):
    """Specification for a workflow step"""
    id: str
    name: str
    icon: str
    status: str  # pending, active, complete, error
    message: Optional[str] = None
    duration: Optional[float] = None


class TableSpec(BaseModel):
    """Specification for a data table"""
    columns: List[Dict[str, Any]]  # {key, label, sortable?, width?}
    rows: List[Dict[str, Any]]
    sortable: bool = False
    filterable: bool = False
    pagination: Optional[Dict[str, int]] = None  # {page, pageSize, total}


class ChartSpec(BaseModel):
    """Specification for charts"""
    chartType: str  # bar, line, pie, donut, radar
    data: List[Dict[str, Any]]
    options: Optional[Dict[str, Any]] = None


# ==================== AG-UI Events ====================

class AGUIEvent(BaseModel):
    """Event sent from agent to UI or vice versa"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: UIEventType
    timestamp: datetime = Field(default_factory=datetime.now)
    source: str  # agent_id or "user"
    target: Optional[str] = None  # component_id or "global"
    payload: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True
    
    def to_sse(self) -> str:
        """Convert to Server-Sent Event format"""
        return f"data: {self.model_dump_json()}\n\n"


class AGUISurface(BaseModel):
    """A rendering surface that contains UI components"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    components: List[UIComponent] = Field(default_factory=list)
    dataModel: Dict[str, Any] = Field(default_factory=dict)


# ==================== Agent UI Builders ====================

class AGUIBuilder:
    """Builder class for creating AG-UI components programmatically"""
    
    @staticmethod
    def agent_card(
        agent_name: str,
        icon: str,
        phase: AgentPhase,
        message: str,
        fixes: Optional[List[FixCardSpec]] = None,
        stats: Optional[Dict[str, Any]] = None,
        progress: Optional[int] = None
    ) -> UIComponent:
        """Create an agent status card"""
        return UIComponent(
            type=UIComponentType.AGENT_STATUS,
            props={
                "agentName": agent_name,
                "icon": icon,
                "phase": phase.value if isinstance(phase, AgentPhase) else phase,
                "message": message,
                "fixes": [f.model_dump() if hasattr(f, 'model_dump') else f for f in (fixes or [])],
                "stats": stats or {},
                "progress": progress
            },
            style=UIStyle(animation="slideIn")
        )
    
    @staticmethod
    def fix_card(
        agent: str,
        description: str,
        severity: Severity,
        before: Optional[str] = None,
        after: Optional[str] = None,
        line: Optional[int] = None,
        category: Optional[str] = None
    ) -> UIComponent:
        """Create a fix card showing what was changed"""
        return UIComponent(
            type=UIComponentType.FIX_CARD,
            props={
                "agent": agent,
                "description": description,
                "severity": severity.value if isinstance(severity, Severity) else severity,
                "before": before,
                "after": after,
                "line": line,
                "category": category,
                "applied": True
            },
            style=UIStyle(animation="fadeIn")
        )
    
    @staticmethod
    def code_diff(
        before: str,
        after: str,
        title: Optional[str] = None,
        language: str = "python"
    ) -> UIComponent:
        """Create a code diff component"""
        return UIComponent(
            type=UIComponentType.CODE_DIFF,
            props={
                "before": before,
                "after": after,
                "title": title,
                "language": language,
                "lineNumbers": True,
                "highlightChanges": True
            }
        )
    
    @staticmethod
    def progress_bar(
        value: int,
        max_value: int = 100,
        label: Optional[str] = None,
        variant: str = "primary"
    ) -> UIComponent:
        """Create a progress bar"""
        return UIComponent(
            type=UIComponentType.PROGRESS,
            props={
                "value": value,
                "max": max_value,
                "label": label
            },
            style=UIStyle(variant=variant)
        )
    
    @staticmethod
    def workflow_timeline(steps: List[WorkflowStepSpec]) -> UIComponent:
        """Create a workflow timeline"""
        return UIComponent(
            type=UIComponentType.TIMELINE,
            props={
                "steps": [s.model_dump() if hasattr(s, 'model_dump') else s for s in steps],
                "orientation": "horizontal"
            }
        )
    
    @staticmethod
    def stat_card(
        label: str,
        value: Union[str, int, float],
        icon: Optional[str] = None,
        change: Optional[str] = None,
        variant: str = "default"
    ) -> UIComponent:
        """Create a statistics card"""
        return UIComponent(
            type=UIComponentType.STAT,
            props={
                "label": label,
                "value": value,
                "icon": icon,
                "change": change
            },
            style=UIStyle(variant=variant)
        )
    
    @staticmethod
    def alert(
        message: str,
        severity: Severity,
        title: Optional[str] = None,
        dismissible: bool = True
    ) -> UIComponent:
        """Create an alert component"""
        return UIComponent(
            type=UIComponentType.ALERT,
            props={
                "message": message,
                "title": title,
                "dismissible": dismissible
            },
            style=UIStyle(variant=severity.value if isinstance(severity, Severity) else severity),
            actions=[UIAction(name="dismiss", icon="x")] if dismissible else None
        )
    
    @staticmethod
    def vulnerability_card(
        vuln_type: str,
        severity: Severity,
        description: str,
        line: Optional[int] = None,
        pattern: Optional[str] = None,
        fix_available: bool = False
    ) -> UIComponent:
        """Create a vulnerability card"""
        return UIComponent(
            type=UIComponentType.VULNERABILITY_CARD,
            props={
                "type": vuln_type,
                "severity": severity.value if isinstance(severity, Severity) else severity,
                "description": description,
                "line": line,
                "pattern": pattern,
                "fixAvailable": fix_available
            },
            style=UIStyle(variant=severity.value if isinstance(severity, Severity) else severity)
        )
    
    @staticmethod
    def data_table(
        columns: List[Dict[str, Any]],
        rows: List[Dict[str, Any]],
        title: Optional[str] = None,
        sortable: bool = True
    ) -> UIComponent:
        """Create a data table"""
        return UIComponent(
            type=UIComponentType.TABLE,
            props={
                "columns": columns,
                "rows": rows,
                "title": title,
                "sortable": sortable
            }
        )
    
    @staticmethod
    def expandable_section(
        title: str,
        children: List[UIComponent],
        expanded: bool = False,
        icon: Optional[str] = None
    ) -> UIComponent:
        """Create an expandable section"""
        return UIComponent(
            type=UIComponentType.EXPANDABLE,
            props={
                "title": title,
                "expanded": expanded,
                "icon": icon
            },
            children=children
        )
    
    @staticmethod
    def button_group(buttons: List[Dict[str, Any]]) -> UIComponent:
        """Create a button group"""
        return UIComponent(
            type=UIComponentType.BUTTON_GROUP,
            props={"buttons": buttons}
        )
    
    @staticmethod
    def card(
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        children: Optional[List[UIComponent]] = None,
        footer: Optional[str] = None,
        variant: str = "default",
        collapsible: bool = False
    ) -> UIComponent:
        """Create a card container"""
        return UIComponent(
            type=UIComponentType.CARD,
            props={
                "title": title,
                "subtitle": subtitle,
                "footer": footer,
                "collapsible": collapsible
            },
            children=children,
            style=UIStyle(variant=variant)
        )
    
    @staticmethod
    def grid(children: List[UIComponent], columns: int = 2, gap: str = "16px") -> UIComponent:
        """Create a grid layout"""
        return UIComponent(
            type=UIComponentType.GRID,
            props={
                "columns": columns,
                "gap": gap
            },
            children=children
        )


# ==================== Event Builders ====================

class AGUIEventBuilder:
    """Builder for creating AG-UI events"""
    
    @staticmethod
    def agent_start(agent_id: str, agent_name: str, message: str) -> AGUIEvent:
        """Create agent start event"""
        return AGUIEvent(
            type=UIEventType.START,
            source=agent_id,
            payload={
                "agentName": agent_name,
                "message": message,
                "ui": AGUIBuilder.agent_card(
                    agent_name=agent_name,
                    icon=get_agent_icon(agent_id),
                    phase=AgentPhase.STARTING,
                    message=message
                ).model_dump()
            }
        )
    
    @staticmethod
    def agent_progress(
        agent_id: str,
        agent_name: str,
        message: str,
        progress: Optional[int] = None,
        phase: AgentPhase = AgentPhase.PROCESSING
    ) -> AGUIEvent:
        """Create agent progress event"""
        return AGUIEvent(
            type=UIEventType.PROGRESS,
            source=agent_id,
            payload={
                "agentName": agent_name,
                "message": message,
                "progress": progress,
                "phase": phase.value,
                "ui": AGUIBuilder.agent_card(
                    agent_name=agent_name,
                    icon=get_agent_icon(agent_id),
                    phase=phase,
                    message=message,
                    progress=progress
                ).model_dump()
            }
        )
    
    @staticmethod
    def agent_complete(
        agent_id: str,
        agent_name: str,
        message: str,
        fixes: Optional[List[FixCardSpec]] = None,
        stats: Optional[Dict[str, Any]] = None,
        duration: Optional[float] = None
    ) -> AGUIEvent:
        """Create agent completion event"""
        return AGUIEvent(
            type=UIEventType.COMPLETE,
            source=agent_id,
            payload={
                "agentName": agent_name,
                "message": message,
                "fixes": [f.model_dump() if hasattr(f, 'model_dump') else f for f in (fixes or [])],
                "stats": stats or {},
                "duration": duration,
                "ui": AGUIBuilder.agent_card(
                    agent_name=agent_name,
                    icon=get_agent_icon(agent_id),
                    phase=AgentPhase.COMPLETE,
                    message=message,
                    fixes=fixes,
                    stats=stats
                ).model_dump()
            }
        )
    
    @staticmethod
    def code_update(
        agent_id: str,
        code: str,
        fixes: List[FixCardSpec],
        original_code: Optional[str] = None
    ) -> AGUIEvent:
        """Create code update event with diff"""
        ui_components = []
        
        # Add fix cards
        for fix in fixes:
            ui_components.append(AGUIBuilder.fix_card(
                agent=fix.agent if hasattr(fix, 'agent') else agent_id,
                description=fix.description if hasattr(fix, 'description') else str(fix),
                severity=fix.severity if hasattr(fix, 'severity') else Severity.MEDIUM,
                before=fix.before if hasattr(fix, 'before') else None,
                after=fix.after if hasattr(fix, 'after') else None,
                line=fix.line if hasattr(fix, 'line') else None
            ).model_dump())
        
        # Add diff if we have original
        if original_code and code != original_code:
            ui_components.append(AGUIBuilder.code_diff(
                before=original_code,
                after=code,
                title="Code Changes"
            ).model_dump())
        
        return AGUIEvent(
            type=UIEventType.UPDATE,
            source=agent_id,
            payload={
                "code": code,
                "fixCount": len(fixes),
                "ui": ui_components
            }
        )
    
    @staticmethod
    def render_ui(components: List[UIComponent], surface_id: str = "main") -> AGUIEvent:
        """Create render event with UI components"""
        return AGUIEvent(
            type=UIEventType.RENDER,
            source="system",
            target=surface_id,
            payload={
                "components": [c.model_dump() if hasattr(c, 'model_dump') else c for c in components]
            }
        )
    
    @staticmethod
    def workflow_update(steps: List[WorkflowStepSpec]) -> AGUIEvent:
        """Create workflow timeline update"""
        return AGUIEvent(
            type=UIEventType.UPDATE,
            source="system",
            target="workflow",
            payload={
                "ui": AGUIBuilder.workflow_timeline(steps).model_dump()
            }
        )
    
    @staticmethod
    def stream_chunk(content: str, chunk_type: str = "code") -> AGUIEvent:
        """Create a stream chunk event"""
        return AGUIEvent(
            type=UIEventType.STREAM_CHUNK,
            source="code_generator",
            payload={
                "content": content,
                "chunkType": chunk_type
            }
        )
    
    @staticmethod
    def stream_end(total_content: str, stats: Optional[Dict[str, Any]] = None) -> AGUIEvent:
        """Create stream end event"""
        return AGUIEvent(
            type=UIEventType.STREAM_END,
            source="code_generator",
            payload={
                "totalContent": total_content,
                "stats": stats or {}
            }
        )
    
    @staticmethod
    def error(source: str, message: str, details: Optional[Dict[str, Any]] = None) -> AGUIEvent:
        """Create an error event"""
        return AGUIEvent(
            type=UIEventType.ERROR,
            source=source,
            payload={
                "message": message,
                "details": details or {},
                "ui": AGUIBuilder.alert(
                    message=message,
                    severity=Severity.HIGH,
                    title="Error"
                ).model_dump()
            }
        )


def get_agent_icon(agent_id: str) -> str:
    """Get icon for agent"""
    icons = {
        "code_generator": "⚡",
        "validator": "✓",
        "testing": "🧪",
        "security": "🛡️"
    }
    return icons.get(agent_id.lower(), "🤖")


def create_fix_spec(
    agent: str,
    description: str,
    severity: str = "medium",
    before: Optional[str] = None,
    after: Optional[str] = None,
    line: Optional[int] = None,
    category: Optional[str] = None
) -> FixCardSpec:
    """Helper to create a FixCardSpec from agent output"""
    severity_map = {
        "critical": Severity.CRITICAL,
        "high": Severity.HIGH,
        "medium": Severity.MEDIUM,
        "low": Severity.LOW,
        "info": Severity.INFO
    }
    return FixCardSpec(
        agent=agent,
        description=description,
        severity=severity_map.get(severity.lower(), Severity.MEDIUM),
        before=before,
        after=after,
        line=line,
        category=category
    )


# ==================== SSE Helpers ====================

def serialize_event(event: AGUIEvent) -> str:
    """Serialize an event for SSE transmission"""
    return event.to_sse()


def serialize_component(component: UIComponent) -> str:
    """Serialize a component to JSON"""
    return component.model_dump_json()
