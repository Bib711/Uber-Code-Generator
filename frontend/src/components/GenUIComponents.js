// AG-UI Component Library - Generative UI Components for Agent Interfaces
// These components render dynamically based on JSON specs from agents

import React, { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './GenUIComponents.css';

// ==================== ANIMATION VARIANTS ====================

const slideIn = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -20 }
};

const slideInLeft = {
  initial: { opacity: 0, x: -30 },
  animate: { opacity: 1, x: 0 },
  exit: { opacity: 0, x: 30 }
};

const scaleIn = {
  initial: { opacity: 0, scale: 0.9 },
  animate: { opacity: 1, scale: 1 },
  exit: { opacity: 0, scale: 0.9 }
};

// ==================== UTILITY COMPONENTS ====================

export const SeverityBadge = ({ severity, showIcon = true }) => {
  const config = {
    critical: { icon: '🔴', label: 'Critical', className: 'severity-critical' },
    high: { icon: '🟠', label: 'High', className: 'severity-high' },
    medium: { icon: '🟡', label: 'Medium', className: 'severity-medium' },
    low: { icon: '🟢', label: 'Low', className: 'severity-low' },
    info: { icon: '🔵', label: 'Info', className: 'severity-info' }
  };
  
  const { icon, label, className } = config[severity?.toLowerCase()] || config.info;
  
  return (
    <span className={`genui-severity-badge ${className}`}>
      {showIcon && <span className="severity-icon">{icon}</span>}
      {label}
    </span>
  );
};

export const StatusIndicator = ({ status, pulse = false }) => {
  const statusConfig = {
    pending: { color: '#6b7280', icon: '○' },
    active: { color: '#3b82f6', icon: '●' },
    complete: { color: '#10b981', icon: '✓' },
    error: { color: '#ef4444', icon: '✕' }
  };
  
  const { color, icon } = statusConfig[status] || statusConfig.pending;
  
  return (
    <span 
      className={`genui-status-indicator ${status} ${pulse ? 'pulse' : ''}`}
      style={{ color }}
    >
      {icon}
    </span>
  );
};

// ==================== LAYOUT COMPONENTS ====================

export const Card = ({ 
  title, 
  subtitle, 
  children, 
  footer, 
  variant = 'default',
  collapsible = false,
  defaultExpanded = true,
  icon,
  actions,
  className = ''
}) => {
  const [expanded, setExpanded] = useState(defaultExpanded);
  
  return (
    <motion.div 
      className={`genui-card ${variant} ${className}`}
      variants={slideIn}
      initial="initial"
      animate="animate"
      exit="exit"
    >
      {(title || actions) && (
        <div className="genui-card-header" onClick={collapsible ? () => setExpanded(!expanded) : undefined}>
          <div className="genui-card-title-wrapper">
            {icon && <span className="genui-card-icon">{icon}</span>}
            <div>
              {title && <h3 className="genui-card-title">{title}</h3>}
              {subtitle && <p className="genui-card-subtitle">{subtitle}</p>}
            </div>
          </div>
          <div className="genui-card-actions">
            {actions}
            {collapsible && (
              <button className="genui-collapse-btn">
                {expanded ? '▼' : '▶'}
              </button>
            )}
          </div>
        </div>
      )}
      <AnimatePresence>
        {expanded && (
          <motion.div 
            className="genui-card-content"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
          >
            {children}
          </motion.div>
        )}
      </AnimatePresence>
      {footer && <div className="genui-card-footer">{footer}</div>}
    </motion.div>
  );
};

export const Grid = ({ children, columns = 2, gap = '16px', className = '' }) => (
  <div 
    className={`genui-grid ${className}`}
    style={{ 
      display: 'grid',
      gridTemplateColumns: `repeat(${columns}, 1fr)`,
      gap 
    }}
  >
    {children}
  </div>
);

export const Divider = ({ text }) => (
  <div className="genui-divider">
    {text && <span>{text}</span>}
  </div>
);

// ==================== DATA DISPLAY COMPONENTS ====================

export const StatCard = ({ label, value, icon, change, variant = 'default', trend }) => {
  const trendClass = trend === 'up' ? 'trend-up' : trend === 'down' ? 'trend-down' : '';
  
  return (
    <motion.div 
      className={`genui-stat-card ${variant}`}
      variants={scaleIn}
      initial="initial"
      animate="animate"
      whileHover={{ scale: 1.02 }}
    >
      <div className="genui-stat-header">
        {icon && <span className="genui-stat-icon">{icon}</span>}
        <span className="genui-stat-label">{label}</span>
      </div>
      <div className="genui-stat-value">{value}</div>
      {change && (
        <div className={`genui-stat-change ${trendClass}`}>
          {change}
        </div>
      )}
    </motion.div>
  );
};

export const ProgressBar = ({ value, max = 100, label, variant = 'primary', showValue = true, animated = true }) => {
  const percentage = Math.min(100, Math.max(0, (value / max) * 100));
  
  return (
    <div className={`genui-progress ${variant}`}>
      {label && <div className="genui-progress-label">{label}</div>}
      <div className="genui-progress-track">
        <motion.div 
          className={`genui-progress-bar ${animated ? 'animated' : ''}`}
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
        />
      </div>
      {showValue && <span className="genui-progress-value">{Math.round(percentage)}%</span>}
    </div>
  );
};

export const Badge = ({ children, variant = 'default', size = 'md', icon }) => (
  <span className={`genui-badge ${variant} ${size}`}>
    {icon && <span className="genui-badge-icon">{icon}</span>}
    {children}
  </span>
);

// ==================== CODE COMPONENTS ====================

export const CodeBlock = ({ code, language = 'python', title, lineNumbers = true, maxHeight = '400px', highlightLines = [] }) => {
  const [copied, setCopied] = useState(false);
  const lines = code?.split('\n') || [];
  
  const copyCode = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  
  return (
    <div className="genui-code-block">
      {title && (
        <div className="genui-code-header">
          <span className="genui-code-title">{title}</span>
          <div className="genui-code-actions">
            <span className="genui-code-language">{language}</span>
            <button className="genui-copy-btn" onClick={copyCode}>
              {copied ? '✓ Copied' : '📋 Copy'}
            </button>
          </div>
        </div>
      )}
      <pre className="genui-code-pre" style={{ maxHeight }}>
        <code className={`language-${language}`}>
          {lines.map((line, i) => (
            <div 
              key={i} 
              className={`genui-code-line ${highlightLines.includes(i + 1) ? 'highlighted' : ''}`}
            >
              {lineNumbers && <span className="genui-line-number">{i + 1}</span>}
              <span className="genui-line-content">{line || ' '}</span>
            </div>
          ))}
        </code>
      </pre>
    </div>
  );
};

export const CodeDiff = ({ before, after, title, language = 'python', expanded: defaultExpanded = true }) => {
  const [expanded, setExpanded] = useState(defaultExpanded);
  const [viewMode, setViewMode] = useState('split'); // 'split' | 'unified'
  
  const beforeLines = before?.split('\n') || [];
  const afterLines = after?.split('\n') || [];
  
  // Simple diff - find changed lines
  const maxLines = Math.max(beforeLines.length, afterLines.length);
  
  return (
    <motion.div 
      className="genui-code-diff"
      variants={slideIn}
      initial="initial"
      animate="animate"
    >
      <div className="genui-diff-header" onClick={() => setExpanded(!expanded)}>
        <div className="genui-diff-title">
          <span className="genui-diff-icon">📝</span>
          {title || 'Code Changes'}
          <Badge variant="primary" size="sm">
            {afterLines.length - beforeLines.length >= 0 ? '+' : ''}{afterLines.length - beforeLines.length} lines
          </Badge>
        </div>
        <div className="genui-diff-actions">
          <button 
            className={`genui-view-btn ${viewMode === 'split' ? 'active' : ''}`}
            onClick={(e) => { e.stopPropagation(); setViewMode('split'); }}
          >
            Split
          </button>
          <button 
            className={`genui-view-btn ${viewMode === 'unified' ? 'active' : ''}`}
            onClick={(e) => { e.stopPropagation(); setViewMode('unified'); }}
          >
            Unified
          </button>
          <span className="genui-expand-icon">{expanded ? '▼' : '▶'}</span>
        </div>
      </div>
      
      <AnimatePresence>
        {expanded && (
          <motion.div 
            className={`genui-diff-content ${viewMode}`}
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
          >
            {viewMode === 'split' ? (
              <div className="genui-diff-split">
                <div className="genui-diff-pane before">
                  <div className="genui-diff-pane-header">Before</div>
                  <pre>
                    {beforeLines.map((line, i) => (
                      <div key={i} className={`genui-diff-line ${afterLines[i] !== line ? 'removed' : ''}`}>
                        <span className="genui-line-number">{i + 1}</span>
                        <span className="genui-line-content">{line || ' '}</span>
                      </div>
                    ))}
                  </pre>
                </div>
                <div className="genui-diff-pane after">
                  <div className="genui-diff-pane-header">After</div>
                  <pre>
                    {afterLines.map((line, i) => (
                      <div key={i} className={`genui-diff-line ${beforeLines[i] !== line ? 'added' : ''}`}>
                        <span className="genui-line-number">{i + 1}</span>
                        <span className="genui-line-content">{line || ' '}</span>
                      </div>
                    ))}
                  </pre>
                </div>
              </div>
            ) : (
              <div className="genui-diff-unified">
                <pre>
                  {Array.from({ length: maxLines }).map((_, i) => {
                    const beforeLine = beforeLines[i];
                    const afterLine = afterLines[i];
                    const isDifferent = beforeLine !== afterLine;
                    
                    return (
                      <React.Fragment key={i}>
                        {isDifferent && beforeLine !== undefined && (
                          <div className="genui-diff-line removed">
                            <span className="genui-diff-symbol">-</span>
                            <span className="genui-line-number">{i + 1}</span>
                            <span className="genui-line-content">{beforeLine}</span>
                          </div>
                        )}
                        {isDifferent && afterLine !== undefined && (
                          <div className="genui-diff-line added">
                            <span className="genui-diff-symbol">+</span>
                            <span className="genui-line-number">{i + 1}</span>
                            <span className="genui-line-content">{afterLine}</span>
                          </div>
                        )}
                        {!isDifferent && beforeLine !== undefined && (
                          <div className="genui-diff-line unchanged">
                            <span className="genui-diff-symbol"> </span>
                            <span className="genui-line-number">{i + 1}</span>
                            <span className="genui-line-content">{beforeLine}</span>
                          </div>
                        )}
                      </React.Fragment>
                    );
                  })}
                </pre>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

// ==================== AGENT-SPECIFIC COMPONENTS ====================

export const AgentStatusCard = ({ 
  agentName, 
  icon, 
  phase, 
  message, 
  progress, 
  fixes = [], 
  stats = {},
  duration,
  variant = 'default'
}) => {
  const phaseConfig = {
    idle: { label: 'Idle', color: '#6b7280' },
    starting: { label: 'Starting', color: '#3b82f6' },
    analyzing: { label: 'Analyzing', color: '#8b5cf6' },
    processing: { label: 'Processing', color: '#f59e0b' },
    fixing: { label: 'Fixing', color: '#ec4899' },
    complete: { label: 'Complete', color: '#10b981' },
    error: { label: 'Error', color: '#ef4444' }
  };
  
  const { label: phaseLabel, color: phaseColor } = phaseConfig[phase] || phaseConfig.idle;
  const isActive = ['starting', 'analyzing', 'processing', 'fixing'].includes(phase);
  
  return (
    <motion.div 
      className={`genui-agent-card ${variant} ${phase}`}
      variants={slideIn}
      initial="initial"
      animate="animate"
      layout
    >
      <div className="genui-agent-header">
        <div className="genui-agent-identity">
          <span className={`genui-agent-icon ${isActive ? 'pulse' : ''}`}>{icon}</span>
          <div className="genui-agent-info">
            <h4 className="genui-agent-name">{agentName}</h4>
            <span className="genui-agent-phase" style={{ color: phaseColor }}>
              {isActive && <span className="genui-spinner" />}
              {phaseLabel}
            </span>
          </div>
        </div>
        {duration !== undefined && (
          <Badge variant="secondary" size="sm" icon="⏱️">
            {duration}s
          </Badge>
        )}
      </div>
      
      <div className="genui-agent-message">{message}</div>
      
      {progress !== undefined && isActive && (
        <ProgressBar value={progress} variant="primary" animated />
      )}
      
      {fixes.length > 0 && (
        <motion.div 
          className="genui-agent-fixes"
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
        >
          <div className="genui-fixes-header">
            <span>🔧 {fixes.length} Fix{fixes.length !== 1 ? 'es' : ''} Applied</span>
          </div>
          <div className="genui-fixes-list">
            {fixes.slice(0, 3).map((fix, i) => (
              <div key={i} className="genui-fix-item">
                <SeverityBadge severity={fix.severity} />
                <span className="genui-fix-desc">
                  {typeof fix === 'object' ? fix.description : fix}
                </span>
              </div>
            ))}
            {fixes.length > 3 && (
              <div className="genui-fixes-more">
                +{fixes.length - 3} more fixes
              </div>
            )}
          </div>
        </motion.div>
      )}
      
      {Object.keys(stats).length > 0 && (
        <div className="genui-agent-stats">
          {Object.entries(stats).map(([key, value]) => (
            <span key={key} className="genui-stat-item">
              <strong>{formatStatKey(key)}:</strong> {value}
            </span>
          ))}
        </div>
      )}
    </motion.div>
  );
};

export const FixCard = ({ 
  agent, 
  description, 
  severity, 
  before, 
  after, 
  line, 
  category,
  applied = true,
  onApply,
  onRevert,
  expandable = true
}) => {
  const [expanded, setExpanded] = useState(false);
  const hasDiff = before && after;
  
  return (
    <motion.div 
      className={`genui-fix-card ${severity} ${applied ? 'applied' : ''}`}
      variants={slideInLeft}
      initial="initial"
      animate="animate"
      layout
    >
      <div 
        className="genui-fix-header" 
        onClick={expandable && hasDiff ? () => setExpanded(!expanded) : undefined}
      >
        <div className="genui-fix-meta">
          <SeverityBadge severity={severity} />
          {category && <Badge variant="secondary" size="sm">{category}</Badge>}
          {line && <Badge variant="secondary" size="sm">Line {line}</Badge>}
        </div>
        <div className="genui-fix-agent">
          {agent}
          {expandable && hasDiff && (
            <span className="genui-expand-icon">{expanded ? '▼' : '▶'}</span>
          )}
        </div>
      </div>
      
      <div className="genui-fix-description">{description}</div>
      
      <AnimatePresence>
        {expanded && hasDiff && (
          <motion.div 
            className="genui-fix-diff"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
          >
            <div className="genui-fix-before">
              <span className="genui-diff-label">Before:</span>
              <code>{before}</code>
            </div>
            <div className="genui-fix-after">
              <span className="genui-diff-label">After:</span>
              <code>{after}</code>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
      {(onApply || onRevert) && (
        <div className="genui-fix-actions">
          {!applied && onApply && (
            <button className="genui-btn primary" onClick={onApply}>
              Apply Fix
            </button>
          )}
          {applied && onRevert && (
            <button className="genui-btn secondary" onClick={onRevert}>
              Revert
            </button>
          )}
        </div>
      )}
    </motion.div>
  );
};

export const VulnerabilityCard = ({ type, severity, description, line, pattern, fixAvailable }) => (
  <motion.div 
    className={`genui-vulnerability-card ${severity}`}
    variants={slideIn}
    initial="initial"
    animate="animate"
  >
    <div className="genui-vuln-header">
      <SeverityBadge severity={severity} />
      <span className="genui-vuln-type">{type}</span>
    </div>
    <div className="genui-vuln-description">{description}</div>
    {pattern && (
      <div className="genui-vuln-pattern">
        <code>{pattern}</code>
      </div>
    )}
    <div className="genui-vuln-footer">
      {line && <Badge variant="secondary">Line {line}</Badge>}
      {fixAvailable && <Badge variant="success" icon="✓">Fix Available</Badge>}
    </div>
  </motion.div>
);

// ==================== TIMELINE/WORKFLOW COMPONENTS ====================

export const WorkflowTimeline = ({ steps, orientation = 'horizontal' }) => (
  <div className={`genui-workflow-timeline ${orientation}`}>
    {steps.map((step, index) => (
      <React.Fragment key={step.id}>
        <motion.div 
          className={`genui-workflow-step ${step.status}`}
          variants={scaleIn}
          initial="initial"
          animate="animate"
          transition={{ delay: index * 0.1 }}
        >
          <div className={`genui-step-icon ${step.status === 'active' ? 'pulse' : ''}`}>
            {step.status === 'complete' ? '✓' : step.icon}
          </div>
          <div className="genui-step-info">
            <span className="genui-step-name">{step.name}</span>
            {step.duration && (
              <span className="genui-step-duration">{step.duration}s</span>
            )}
          </div>
          {step.status === 'active' && <div className="genui-step-spinner" />}
        </motion.div>
        {index < steps.length - 1 && (
          <div className={`genui-step-connector ${steps[index + 1]?.status !== 'pending' ? 'active' : ''}`} />
        )}
      </React.Fragment>
    ))}
  </div>
);

export const Timeline = ({ items, animated = true }) => (
  <div className="genui-timeline">
    {items.map((item, index) => (
      <motion.div 
        key={item.id || index}
        className={`genui-timeline-item ${item.status || ''}`}
        variants={animated ? slideInLeft : undefined}
        initial={animated ? "initial" : undefined}
        animate={animated ? "animate" : undefined}
        transition={{ delay: index * 0.1 }}
      >
        <div className="genui-timeline-marker">
          {item.icon || <StatusIndicator status={item.status} />}
        </div>
        <div className="genui-timeline-content">
          <h4>{item.title}</h4>
          {item.description && <p>{item.description}</p>}
          {item.timestamp && <span className="genui-timeline-time">{item.timestamp}</span>}
        </div>
      </motion.div>
    ))}
  </div>
);

// ==================== INTERACTIVE COMPONENTS ====================

export const Expandable = ({ title, children, expanded: defaultExpanded = false, icon }) => {
  const [expanded, setExpanded] = useState(defaultExpanded);
  
  return (
    <div className="genui-expandable">
      <button className="genui-expandable-header" onClick={() => setExpanded(!expanded)}>
        {icon && <span className="genui-expandable-icon">{icon}</span>}
        <span className="genui-expandable-title">{title}</span>
        <span className="genui-expand-arrow">{expanded ? '▼' : '▶'}</span>
      </button>
      <AnimatePresence>
        {expanded && (
          <motion.div 
            className="genui-expandable-content"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
          >
            {children}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export const Tabs = ({ tabs, activeTab, onChange }) => {
  const [active, setActive] = useState(activeTab || tabs[0]?.id);
  
  const handleChange = (tabId) => {
    setActive(tabId);
    onChange?.(tabId);
  };
  
  const activeContent = tabs.find(t => t.id === active)?.content;
  
  return (
    <div className="genui-tabs">
      <div className="genui-tabs-header">
        {tabs.map(tab => (
          <button
            key={tab.id}
            className={`genui-tab ${active === tab.id ? 'active' : ''}`}
            onClick={() => handleChange(tab.id)}
          >
            {tab.icon && <span className="genui-tab-icon">{tab.icon}</span>}
            {tab.label}
            {tab.badge !== undefined && (
              <Badge variant="primary" size="sm">{tab.badge}</Badge>
            )}
          </button>
        ))}
      </div>
      <div className="genui-tabs-content">
        <AnimatePresence mode="wait">
          <motion.div
            key={active}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
          >
            {activeContent}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
};

export const DataTable = ({ columns, rows, title, sortable = true, onRowClick }) => {
  const [sortKey, setSortKey] = useState(null);
  const [sortOrder, setSortOrder] = useState('asc');
  
  const sortedRows = useMemo(() => {
    if (!sortKey) return rows;
    return [...rows].sort((a, b) => {
      const aVal = a[sortKey];
      const bVal = b[sortKey];
      const order = sortOrder === 'asc' ? 1 : -1;
      if (aVal < bVal) return -order;
      if (aVal > bVal) return order;
      return 0;
    });
  }, [rows, sortKey, sortOrder]);
  
  const handleSort = (key) => {
    if (sortKey === key) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(key);
      setSortOrder('asc');
    }
  };
  
  return (
    <div className="genui-table-wrapper">
      {title && <h3 className="genui-table-title">{title}</h3>}
      <table className="genui-table">
        <thead>
          <tr>
            {columns.map(col => (
              <th 
                key={col.key}
                className={sortable ? 'sortable' : ''}
                onClick={sortable ? () => handleSort(col.key) : undefined}
                style={{ width: col.width }}
              >
                {col.label}
                {sortKey === col.key && (
                  <span className="genui-sort-indicator">
                    {sortOrder === 'asc' ? '↑' : '↓'}
                  </span>
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sortedRows.map((row, i) => (
            <motion.tr 
              key={row.id || i}
              onClick={() => onRowClick?.(row)}
              className={onRowClick ? 'clickable' : ''}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: i * 0.05 }}
            >
              {columns.map(col => (
                <td key={col.key}>{row[col.key]}</td>
              ))}
            </motion.tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

// ==================== ALERT/NOTIFICATION COMPONENTS ====================

export const Alert = ({ message, title, severity = 'info', dismissible = true, onDismiss }) => {
  const [visible, setVisible] = useState(true);
  
  if (!visible) return null;
  
  const icons = {
    info: 'ℹ️',
    success: '✅',
    warning: '⚠️',
    error: '❌',
    critical: '🚨'
  };
  
  return (
    <motion.div 
      className={`genui-alert ${severity}`}
      variants={slideIn}
      initial="initial"
      animate="animate"
      exit="exit"
    >
      <span className="genui-alert-icon">{icons[severity]}</span>
      <div className="genui-alert-content">
        {title && <strong className="genui-alert-title">{title}</strong>}
        <span className="genui-alert-message">{message}</span>
      </div>
      {dismissible && (
        <button 
          className="genui-alert-dismiss"
          onClick={() => { setVisible(false); onDismiss?.(); }}
        >
          ✕
        </button>
      )}
    </motion.div>
  );
};

// ==================== UTILITY FUNCTIONS ====================

function formatStatKey(key) {
  return key
    .replace(/([A-Z])/g, ' $1')
    .replace(/^./, str => str.toUpperCase())
    .trim();
}

// ==================== COMPONENT EXPORTS ====================

const GenUIComponents = {
  // Layout
  Card,
  Grid,
  Divider,
  
  // Data Display
  StatCard,
  ProgressBar,
  Badge,
  SeverityBadge,
  StatusIndicator,
  
  // Code
  CodeBlock,
  CodeDiff,
  
  // Agent Specific
  AgentStatusCard,
  FixCard,
  VulnerabilityCard,
  
  // Timeline/Workflow
  WorkflowTimeline,
  Timeline,
  
  // Interactive
  Expandable,
  Tabs,
  DataTable,
  
  // Alerts
  Alert
};

export default GenUIComponents;
