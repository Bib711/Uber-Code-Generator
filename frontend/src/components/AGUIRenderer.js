// AG-UI Renderer - Converts AG-UI Protocol events to React components
// This is the bridge between backend agent events and frontend UI rendering

import React from 'react';
import {
  Card,
  Grid,
  StatCard,
  ProgressBar,
  Badge,
  CodeBlock,
  CodeDiff,
  AgentStatusCard,
  FixCard,
  VulnerabilityCard,
  WorkflowTimeline,
  Timeline,
  Expandable,
  Tabs,
  DataTable,
  Alert
} from './GenUIComponents';

// ==================== COMPONENT TYPE REGISTRY ====================

const componentRegistry = {
  // Layout
  card: Card,
  grid: Grid,
  
  // Data Display
  stat: StatCard,
  progress: ProgressBar,
  badge: Badge,
  
  // Code
  code_block: CodeBlock,
  code_diff: CodeDiff,
  
  // Agent Specific
  agent_status: AgentStatusCard,
  fix_card: FixCard,
  vulnerability_card: VulnerabilityCard,
  
  // Timeline/Workflow
  timeline: Timeline,
  workflow_graph: WorkflowTimeline,
  
  // Interactive
  expandable: Expandable,
  tabs: Tabs,
  table: DataTable,
  
  // Alerts
  alert: Alert
};

// ==================== MAIN RENDERER ====================

/**
 * Renders a UI component from AG-UI spec
 * @param {Object} spec - The component specification from the backend
 * @param {Function} onAction - Callback for handling user actions
 * @returns {React.ReactElement|null}
 */
export const renderUIComponent = (spec, onAction) => {
  if (!spec || !spec.type) return null;
  
  const Component = componentRegistry[spec.type];
  if (!Component) {
    console.warn(`Unknown component type: ${spec.type}`);
    return null;
  }
  
  // Extract props from the spec
  const props = {
    ...spec.props,
    key: spec.id,
    className: spec.style?.className,
    variant: spec.style?.variant
  };
  
  // Handle children recursively
  if (spec.children && Array.isArray(spec.children)) {
    props.children = spec.children.map(child => renderUIComponent(child, onAction));
  }
  
  // Handle actions
  if (spec.actions && onAction) {
    spec.actions.forEach(action => {
      const handlerName = `on${action.name.charAt(0).toUpperCase() + action.name.slice(1)}`;
      props[handlerName] = () => onAction(action.name, action.payload);
    });
  }
  
  return <Component {...props} />;
};

/**
 * Renders multiple UI components from an array of specs
 */
export const renderUIComponents = (specs, onAction) => {
  if (!specs || !Array.isArray(specs)) return null;
  return specs.map(spec => renderUIComponent(spec, onAction));
};

// ==================== EVENT PROCESSORS ====================

/**
 * Process an AG-UI event and return render data
 */
export const processAGUIEvent = (event) => {
  const { type, source, payload } = event;
  
  switch (type) {
    case 'start':
      return {
        renderType: 'agent_start',
        agentId: source,
        agentName: payload?.agentName,
        message: payload?.message,
        ui: payload?.ui
      };
      
    case 'progress':
      return {
        renderType: 'agent_progress',
        agentId: source,
        agentName: payload?.agentName,
        message: payload?.message,
        progress: payload?.progress,
        phase: payload?.phase,
        ui: payload?.ui
      };
      
    case 'complete':
      return {
        renderType: 'agent_complete',
        agentId: source,
        agentName: payload?.agentName,
        message: payload?.message,
        fixes: payload?.fixes,
        stats: payload?.stats,
        duration: payload?.duration,
        ui: payload?.ui
      };
      
    case 'stream_chunk':
      return {
        renderType: 'stream',
        content: payload?.content,
        totalLines: payload?.totalLines,
        totalChars: payload?.totalChars
      };
      
    case 'update':
    case 'code_update':
      return {
        renderType: 'code_update',
        source: source,
        code: payload?.code,
        fixCount: payload?.fixCount,
        fixes: payload?.fixes,
        ui: payload?.ui
      };
      
    case 'agent_result':
      return {
        renderType: 'agent_result',
        source: payload?.source || source,
        agentName: payload?.agentName,
        data: payload?.data,
        fixes: payload?.fixes,
        stats: payload?.stats,
        ui: payload?.ui
      };
      
    case 'workflow_update':
      return {
        renderType: 'workflow',
        steps: payload?.steps || event.steps
      };
      
    case 'error':
      return {
        renderType: 'error',
        source: source,
        message: payload?.message,
        details: payload?.details,
        ui: payload?.ui
      };
      
    default:
      return { renderType: 'unknown', event };
  }
};

// ==================== SPECIALIZED RENDERERS ====================

/**
 * Render an agent status based on processed event
 */
export const renderAgentStatus = (processed, existingAgents = {}) => {
  const { agentId, agentName, message, progress, phase, fixes, stats, duration, ui } = processed;
  
  // If we have a pre-built UI spec, use it
  if (ui) {
    return renderUIComponent(ui);
  }
  
  // Otherwise, build the component
  const icon = getAgentIcon(agentId);
  
  return (
    <AgentStatusCard
      key={agentId}
      agentName={agentName || agentId}
      icon={icon}
      phase={phase || (processed.renderType === 'agent_complete' ? 'complete' : 'processing')}
      message={message}
      progress={progress}
      fixes={fixes}
      stats={stats}
      duration={duration}
    />
  );
};

/**
 * Render fixes from an agent
 */
export const renderFixes = (fixes, agentName) => {
  if (!fixes || fixes.length === 0) return null;
  
  return (
    <div className="genui-fixes-container">
      <h4 className="genui-fixes-title">
        🔧 {fixes.length} Fix{fixes.length !== 1 ? 'es' : ''} by {agentName}
      </h4>
      {fixes.map((fix, index) => (
        <FixCard
          key={`${agentName}-fix-${index}`}
          agent={fix.agent || agentName}
          description={fix.description || (typeof fix === 'string' ? fix : JSON.stringify(fix))}
          severity={fix.severity || 'medium'}
          before={fix.before}
          after={fix.after}
          line={fix.line}
          category={fix.category}
          applied={true}
        />
      ))}
    </div>
  );
};

/**
 * Render workflow timeline
 */
export const renderWorkflow = (steps) => {
  if (!steps || steps.length === 0) return null;
  
  return <WorkflowTimeline steps={steps} orientation="horizontal" />;
};

/**
 * Render completion summary
 */
export const renderCompletionSummary = (data) => {
  const { stats, total_fixes, ui } = data;
  
  // If we have pre-built UI components, render them
  if (ui?.components) {
    return (
      <div className="genui-completion-summary">
        {renderUIComponents(ui.components)}
      </div>
    );
  }
  
  return (
    <div className="genui-completion-summary">
      <Alert 
        severity="success" 
        title="Generation Complete"
        message={ui?.message || `Generated successfully with ${total_fixes} fixes applied`}
        dismissible={false}
      />
      <Grid columns={4}>
        <StatCard label="Lines" value={stats?.totalLines || '-'} icon="📝" />
        <StatCard label="Fixes" value={stats?.totalFixes || 0} icon="🔧" />
        <StatCard label="Duration" value={`${stats?.totalDuration || '-'}s`} icon="⏱️" />
        <StatCard label="Agents" value={4} icon="🤖" />
      </Grid>
    </div>
  );
};

// ==================== UTILITY FUNCTIONS ====================

function getAgentIcon(agentId) {
  const icons = {
    code_generator: '⚡',
    validator: '✓',
    testing: '🧪',
    security: '🛡️'
  };
  return icons[agentId?.toLowerCase()] || '🤖';
}

/**
 * Parse SSE data line
 */
export const parseSSEData = (line) => {
  if (!line.startsWith('data: ')) return null;
  
  try {
    return JSON.parse(line.slice(6));
  } catch (e) {
    console.warn('Failed to parse SSE data:', e);
    return null;
  }
};

/**
 * Create a state manager for tracking agent states during streaming
 */
export const createAgentStateManager = () => {
  let agents = {};
  let workflow = [];
  let code = '';
  let originalCode = '';
  let allFixes = [];
  
  return {
    updateAgent: (agentId, data) => {
      agents[agentId] = { ...agents[agentId], ...data };
      return { ...agents };
    },
    
    getAgents: () => ({ ...agents }),
    
    updateWorkflow: (steps) => {
      workflow = steps;
      return [...workflow];
    },
    
    getWorkflow: () => [...workflow],
    
    updateCode: (newCode, isOriginal = false) => {
      if (isOriginal) originalCode = newCode;
      code = newCode;
      return code;
    },
    
    getCode: () => code,
    getOriginalCode: () => originalCode,
    
    addFixes: (agentName, fixes) => {
      allFixes.push({ agent: agentName, fixes });
      return [...allFixes];
    },
    
    getAllFixes: () => [...allFixes],
    
    reset: () => {
      agents = {};
      workflow = [];
      code = '';
      originalCode = '';
      allFixes = [];
    }
  };
};

// ==================== REACT HOOK ====================

/**
 * Hook for managing AG-UI state during streaming
 */
export const useAGUIState = () => {
  const [agents, setAgents] = React.useState({});
  const [workflow, setWorkflow] = React.useState([]);
  const [code, setCode] = React.useState('');
  const [originalCode, setOriginalCode] = React.useState('');
  const [fixes, setFixes] = React.useState([]);
  const [isComplete, setIsComplete] = React.useState(false);
  const [error, setError] = React.useState(null);
  
  const processEvent = React.useCallback((event) => {
    const processed = processAGUIEvent(event);
    
    switch (processed.renderType) {
      case 'agent_start':
      case 'agent_progress':
        setAgents(prev => ({
          ...prev,
          [processed.agentId]: {
            ...prev[processed.agentId],
            ...processed
          }
        }));
        break;
        
      case 'agent_complete':
        setAgents(prev => ({
          ...prev,
          [processed.agentId]: {
            ...prev[processed.agentId],
            ...processed,
            phase: 'complete'
          }
        }));
        break;
        
      case 'stream':
        setCode(prev => prev + processed.content);
        break;
        
      case 'code_update':
        setCode(processed.code);
        if (processed.fixes) {
          setFixes(prev => [...prev, { agent: processed.source, fixes: processed.fixes }]);
        }
        break;
        
      case 'workflow':
        setWorkflow(processed.steps);
        break;
        
      case 'error':
        setError(processed);
        break;
        
      default:
        break;
    }
    
    // Check for completion
    if (event.type === 'complete') {
      setIsComplete(true);
      if (event.payload?.original_code) {
        setOriginalCode(event.payload.original_code);
      }
    }
    
    return processed;
  }, []);
  
  const reset = React.useCallback(() => {
    setAgents({});
    setWorkflow([]);
    setCode('');
    setOriginalCode('');
    setFixes([]);
    setIsComplete(false);
    setError(null);
  }, []);
  
  return {
    agents,
    workflow,
    code,
    originalCode,
    fixes,
    isComplete,
    error,
    processEvent,
    reset
  };
};

// ==================== EXPORTS ====================

const AGUIRenderer = {
  renderUIComponent,
  renderUIComponents,
  processAGUIEvent,
  renderAgentStatus,
  renderFixes,
  renderWorkflow,
  renderCompletionSummary,
  parseSSEData,
  createAgentStateManager,
  useAGUIState
};

export default AGUIRenderer;
