// Generative UI Chat Interface V2 - Clean Split-Panel Design
// Professional-grade, spacious layout with AG-UI Protocol

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import UserMenu from '../components/UserMenu';
import { motion, AnimatePresence } from 'framer-motion';
import {
  CodeBlock,
  CodeDiff,
  FixCard,
  Expandable,
  Tabs,
  Badge
} from '../components/GenUIComponents';
import { useAGUIState, parseSSEData } from '../components/AGUIRenderer';
import '../components/GenUIComponents.css';
import './GenUIChatPageV2.css';

const API_BASE = 'http://localhost:5000/api';

const GenUIChatPageV2 = () => {
  const { sessionId: urlSessionId } = useParams();
  const navigate = useNavigate();
  const { authAxios, token } = useAuth();
  
  // Session state
  const [sessionId, setSessionId] = useState(urlSessionId || null);
  const [sessionTitle, setSessionTitle] = useState('New Chat');
  const [messages, setMessages] = useState([]);
  
  // Input state
  const [prompt, setPrompt] = useState('');
  
  // API key state
  const [apiKey, setApiKey] = useState(localStorage.getItem('groq_api_key') || '');
  const [showSettings, setShowSettings] = useState(false);
  
  // Streaming & result state
  const [loading, setLoading] = useState(false);
  const [streamingCode, setStreamingCode] = useState('');
  const [streamingStats, setStreamingStats] = useState({ lines: 0, chars: 0 });
  
  // AG-UI State
  const { processEvent, reset: resetAGUI } = useAGUIState();
  
  // Result state
  const [result, setResult] = useState(null);
  const [showOriginal, setShowOriginal] = useState(false);
  
  // Edit message state
  const [editingMessageIndex, setEditingMessageIndex] = useState(null);
  const [editingContent, setEditingContent] = useState('');
  
  // Panel state
  const [rightPanelTab, setRightPanelTab] = useState('code'); // 'code', 'agents', 'fixes'
  const [showRightPanel, setShowRightPanel] = useState(true);
  const [panelWidth, setPanelWidth] = useState(500);
  const [isResizing, setIsResizing] = useState(false);
  
  // Active agent for display
  const [activeAgent, setActiveAgent] = useState(null);
  const [agentMessages, setAgentMessages] = useState([]);
  
  // Refs
  const messagesEndRef = useRef(null);
  const sessionCreatedRef = useRef(false);
  const resizeRef = useRef(null);
  
  // Scroll to bottom
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);
  
  // Panel resize handlers
  const startResize = useCallback((e) => {
    setIsResizing(true);
    resizeRef.current = e.clientX;
  }, []);
  
  const doResize = useCallback((e) => {
    if (!isResizing) return;
    const diff = resizeRef.current - e.clientX;
    const newWidth = Math.min(Math.max(panelWidth + diff, 350), 900);
    setPanelWidth(newWidth);
    resizeRef.current = e.clientX;
  }, [isResizing, panelWidth]);
  
  const stopResize = useCallback(() => {
    setIsResizing(false);
  }, []);
  
  // Add/remove resize event listeners
  useEffect(() => {
    if (isResizing) {
      window.addEventListener('mousemove', doResize);
      window.addEventListener('mouseup', stopResize);
    }
    return () => {
      window.removeEventListener('mousemove', doResize);
      window.removeEventListener('mouseup', stopResize);
    };
  }, [isResizing, doResize, stopResize]);
  
  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);
  
  // Handle URL session ID changes
  useEffect(() => {
    if (urlSessionId) {
      setSessionId(urlSessionId);
      sessionCreatedRef.current = true;
    } else if (!sessionCreatedRef.current) {
      sessionCreatedRef.current = true;
      createSession();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [urlSessionId]);
  
  // Load session data
  useEffect(() => {
    if (sessionId && token) {
      loadSession();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, token]);
  
  const loadSession = async () => {
    if (!sessionId || !token) return;
    try {
      const response = await authAxios.get(`/sessions/${sessionId}`);
      setSessionTitle(response.data.title || 'New Chat');
      if (response.data.messages?.length > 0) {
        const loadedMessages = response.data.messages.map(msg => ({
          role: msg.role,
          content: msg.content,
          code_output: msg.code_output,
          workflow_data: msg.workflow_data,
          hasResult: !!msg.code_output
        }));
        setMessages(loadedMessages);
        
        // Find last assistant message with code and restore full result
        const lastAssistantMsg = [...response.data.messages].reverse().find(m => m.code_output);
        if (lastAssistantMsg) {
          const workflowData = lastAssistantMsg.workflow_data || {};
          
          // Ensure stats has proper defaults
          const restoredStats = workflowData.stats || {};
          
          setResult({
            code: lastAssistantMsg.code_output,
            original_code: workflowData.original_code,
            all_fixes: workflowData.all_fixes || [],
            code_was_fixed: workflowData.code_was_fixed,
            total_fixes: workflowData.total_fixes || 0,
            validation: workflowData.validation,
            tests: workflowData.tests,
            security: workflowData.security,
            stats: {
              totalDuration: restoredStats.totalDuration || restoredStats.total_duration || null,
              totalLines: restoredStats.totalLines || restoredStats.total_lines || 0,
              totalFixes: restoredStats.totalFixes || restoredStats.total_fixes || 0
            }
          });
          
          // Restore agent messages if available
          if (workflowData.agent_messages) {
            setAgentMessages(workflowData.agent_messages);
          }
        }
      }
    } catch (error) {
      console.error('Error loading session:', error);
    }
  };
  
  const createSession = async () => {
    try {
      const response = await authAxios.post('/sessions');
      const newSessionId = response.data.session_id;
      setSessionId(newSessionId);
      setSessionTitle('New Chat');
      setMessages([]);
      setPrompt('');
      setResult(null);
      setStreamingCode('');
      resetAGUI();
      sessionCreatedRef.current = true;
      navigate(`/chat/${newSessionId}`);
    } catch (error) {
      console.error('Error creating session:', error);
    }
  };
  
  const saveApiKey = () => {
    localStorage.setItem('groq_api_key', apiKey);
    setShowSettings(false);
  };
  
  // Main submit handler
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!prompt.trim() || loading) return;
    
    const userMessage = { role: 'user', content: prompt };
    setMessages(prev => [...prev, userMessage]);
    
    if (sessionId && token) {
      try {
        const saveResponse = await authAxios.post(`/sessions/${sessionId}/messages`, {
          role: 'user',
          content: prompt
        });
        if (saveResponse.data.new_title) {
          setSessionTitle(saveResponse.data.new_title);
        }
      } catch (error) {
        console.error('Error saving user message:', error);
      }
    }
    
    setLoading(true);
    setStreamingCode('');
    setStreamingStats({ lines: 0, chars: 0 });
    setResult(null);
    setShowOriginal(false);
    setAgentMessages([]);
    setShowRightPanel(true);
    setRightPanelTab('code');
    resetAGUI();
    
    try {
      // Include context_code if we have previously generated code
      const requestBody = { 
        prompt, 
        api_key: apiKey,
        context_code: result?.code || null  // Send previous code for follow-up prompts
      };
      
      const response = await fetch(`${API_BASE}/generate/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody)
      });
      
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let fullCode = '';
      let originalCode = ''; // Store the first generated code before any fixes
      let agentResults = {};
      let finalAllFixes = [];
      let localAgentMessages = []; // Track agent messages locally for saving
      const startTime = Date.now(); // Track start time for duration calculation
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          
          const data = parseSSEData(line);
          if (!data) continue;
          
          processEvent(data);
          
          switch (data.type) {
            case 'start':
              setActiveAgent(data.source || data.payload?.agentName);
              const startMsg = {
                agent: data.source || data.payload?.agentName,
                message: data.payload?.message || `Starting ${data.payload?.agentName}...`,
                type: 'start',
                time: new Date().toLocaleTimeString()
              };
              localAgentMessages.push(startMsg);
              setAgentMessages(prev => [...prev, startMsg]);
              break;
              
            case 'stream_chunk':
              fullCode += data.payload?.content || '';
              setStreamingCode(fullCode);
              setStreamingStats({
                lines: data.payload?.totalLines || fullCode.split('\n').length,
                chars: data.payload?.totalChars || fullCode.length
              });
              break;
              
            case 'progress':
              const progressMsg = {
                agent: data.source || data.payload?.agentName,
                message: data.payload?.message,
                progress: data.payload?.progress,
                type: 'progress',
                time: new Date().toLocaleTimeString()
              };
              localAgentMessages.push(progressMsg);
              setAgentMessages(prev => [...prev, progressMsg]);
              break;
              
            case 'code_update':
              // Capture original code before first fix is applied
              if (!originalCode && fullCode) {
                originalCode = fullCode;
              }
              fullCode = data.payload?.code || fullCode;
              setStreamingCode(fullCode);
              if (data.payload?.fixes) {
                finalAllFixes.push({
                  agent: data.source || data.payload?.ui?.agent,
                  fixes: data.payload?.fixes
                });
              }
              const codeUpdateMsg = {
                agent: data.source,
                message: `Applied ${data.payload?.fixCount || 0} fixes`,
                fixes: data.payload?.fixes,
                type: 'code_update',
                time: new Date().toLocaleTimeString()
              };
              localAgentMessages.push(codeUpdateMsg);
              setAgentMessages(prev => [...prev, codeUpdateMsg]);
              break;
              
            case 'agent_result':
              const agentKey = data.payload?.agentName?.toLowerCase() || data.source;
              agentResults[agentKey] = data.payload?.data;
              
              // Capture fixes from agent_result if present
              if (data.payload?.fixes && data.payload.fixes.length > 0) {
                finalAllFixes.push({
                  agent: data.payload?.agentName || data.source,
                  fixes: data.payload.fixes
                });
              }
              
              const resultMsg = {
                agent: data.payload?.agentName || data.source,
                message: data.payload?.stats?.fixesApplied 
                  ? `Completed with ${data.payload.stats.fixesApplied} fixes`
                  : `Completed`,
                stats: data.payload?.stats,
                fixes: data.payload?.fixes,
                type: 'result',
                time: new Date().toLocaleTimeString()
              };
              localAgentMessages.push(resultMsg);
              setAgentMessages(prev => [...prev, resultMsg]);
              break;
              
            case 'complete':
              // Merge backend all_fixes with locally collected fixes
              const backendFixes = data.payload?.all_fixes || [];
              const mergedFixes = backendFixes.length > 0 ? backendFixes : finalAllFixes;
              
              // Calculate duration
              const duration = data.payload?.stats?.totalDuration || ((Date.now() - startTime) / 1000).toFixed(1);
              
              // Use our captured originalCode, fallback to backend's original_code
              const finalOriginalCode = originalCode || data.payload?.original_code || null;
              
              const finalResult = {
                code: data.payload?.code || fullCode,
                original_code: finalOriginalCode,
                prompt: data.payload?.prompt,
                validation: agentResults.validator || data.payload?.validation,
                tests: agentResults.testing || agentResults['testing agent'] || data.payload?.tests,
                security: agentResults.security || agentResults['security agent'] || data.payload?.security,
                all_fixes: mergedFixes,
                code_was_fixed: data.payload?.code_was_fixed || mergedFixes.length > 0,
                total_fixes: data.payload?.total_fixes || mergedFixes.reduce((sum, f) => sum + (f.fixes?.length || 0), 0),
                workflow: data.payload?.workflow,
                stats: {
                  ...data.payload?.stats,
                  totalDuration: duration
                }
              };
              setResult(finalResult);
              setActiveAgent(null);
              break;
              
            default:
              break;
          }
        }
      }
      
      const totalFixCount = finalAllFixes.reduce((sum, f) => sum + (f.fixes?.length || 0), 0);
      const totalDuration = ((Date.now() - startTime) / 1000).toFixed(1);
      const assistantContent = `Generated ${fullCode.split('\n').length} lines of code${totalFixCount > 0 ? ` with ${totalFixCount} auto-fixes` : ''} in ${totalDuration}s`;
      
      // Build final result for state
      // Use captured originalCode (code before any fixes were applied)
      const finalOriginalCodeForSave = originalCode || result?.original_code || null;
      const savedResult = {
        code: fullCode,
        original_code: finalOriginalCodeForSave,
        all_fixes: finalAllFixes,
        code_was_fixed: totalFixCount > 0 || (originalCode && originalCode !== fullCode),
        total_fixes: totalFixCount,
        validation: agentResults.validator,
        tests: agentResults.testing || agentResults['testing agent'],
        security: agentResults.security || agentResults['security agent'],
        stats: {
          totalDuration: totalDuration,
          totalLines: fullCode.split('\n').length,
          totalFixes: totalFixCount
        }
      };
      
      // Update the result state with final values including original_code
      setResult(savedResult);
      
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: assistantContent,
        hasResult: true
      }]);
      
      if (sessionId && token) {
        try {
          // Build stats object for saving
          const statsToSave = {
            totalDuration: totalDuration,
            totalLines: fullCode.split('\n').length,
            totalFixes: totalFixCount
          };
          
          // Save message with full workflow_data including fixes
          await authAxios.post(`/sessions/${sessionId}/messages`, {
            role: 'assistant',
            content: assistantContent,
            code_output: fullCode,
            workflow_data: {
              original_code: savedResult.original_code,
              all_fixes: finalAllFixes,
              code_was_fixed: savedResult.code_was_fixed,
              total_fixes: totalFixCount,
              validation: savedResult.validation,
              tests: savedResult.tests,
              security: savedResult.security,
              stats: statsToSave,
              agent_messages: localAgentMessages.slice(-20) // Save last 20 agent messages
            }
          });
        } catch (error) {
          console.error('Error saving assistant message:', error);
        }
      }
      
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: '❌ Error generating code. Please try again.'
      }]);
    } finally {
      setLoading(false);
      setPrompt('');
    }
  };
  
  const [copyStatus, setCopyStatus] = useState('');
  
  const copyCode = async () => {
    const codeToCopy = showOriginal ? result?.original_code : (result?.code || streamingCode);
    if (!codeToCopy) {
      setCopyStatus('No code');
      setTimeout(() => setCopyStatus(''), 2000);
      return;
    }
    
    // Try multiple methods to copy
    let success = false;
    
    // Method 1: Modern Clipboard API
    if (navigator.clipboard && window.isSecureContext) {
      try {
        await navigator.clipboard.writeText(codeToCopy);
        success = true;
      } catch (err) {
        console.log('Clipboard API failed:', err);
      }
    }
    
    // Method 2: execCommand fallback
    if (!success) {
      const textArea = document.createElement('textarea');
      textArea.value = codeToCopy;
      textArea.style.position = 'fixed';
      textArea.style.top = '0';
      textArea.style.left = '0';
      textArea.style.width = '2em';
      textArea.style.height = '2em';
      textArea.style.padding = '0';
      textArea.style.border = 'none';
      textArea.style.outline = 'none';
      textArea.style.boxShadow = 'none';
      textArea.style.background = 'transparent';
      document.body.appendChild(textArea);
      textArea.focus();
      textArea.select();
      
      try {
        success = document.execCommand('copy');
      } catch (err) {
        console.log('execCommand failed:', err);
      }
      
      document.body.removeChild(textArea);
    }
    
    if (success) {
      setCopyStatus('✓ Copied!');
    } else {
      // If all else fails, show code in alert for manual copy
      setCopyStatus('Select & Copy');
      window.prompt('Copy this code (Ctrl+C):', codeToCopy.substring(0, 500) + (codeToCopy.length > 500 ? '...' : ''));
    }
    setTimeout(() => setCopyStatus(''), 2000);
  };
  
  const getAgentIcon = (agentId) => {
    const icons = {
      code_generator: '⚡',
      validator: '✓',
      testing: '🧪',
      security: '🛡️'
    };
    const key = agentId?.toLowerCase().replace(' agent', '').replace(' ', '_');
    return icons[key] || '🤖';
  };
  
  // Handle clicking on a user message to edit
  const handleEditMessage = (index, content) => {
    if (loading) return;
    setEditingMessageIndex(index);
    setEditingContent(content);
  };
  
  // Handle saving edited message and resubmitting
  const handleSaveEdit = async (e) => {
    e.preventDefault();
    if (!editingContent.trim() || loading) return;
    
    // Remove all messages from this point onwards
    setMessages(prev => prev.slice(0, editingMessageIndex));
    
    // Reset results
    setResult(null);
    setStreamingCode('');
    setAgentMessages([]);
    
    // Close edit mode
    setEditingMessageIndex(null);
    
    // Set prompt and trigger submit
    const newPrompt = editingContent;
    setEditingContent('');
    setPrompt(newPrompt);
    
    // Trigger form submission after state updates
    setTimeout(() => {
      document.querySelector('.input-form')?.requestSubmit();
    }, 50);
  };
  
  // Cancel editing
  const handleCancelEdit = () => {
    setEditingMessageIndex(null);
    setEditingContent('');
  };
  
  // Generate a rich description for the code (ChatGPT style)
  const generateDescription = (code, prompt) => {
    if (!code) return null;
    
    const lines = code.split('\n').length;
    const hasClass = code.includes('class ');
    const hasAsync = code.includes('async ');
    const imports = code.match(/^import |^from /gm)?.length || 0;
    
    // Extract function/class names
    const classMatch = code.match(/class\s+(\w+)/g)?.map(c => c.replace('class ', '')) || [];
    const funcMatch = code.match(/def\s+(\w+)/g)?.map(f => f.replace('def ', '')).filter(f => !f.startsWith('_')) || [];
    const publicMethods = funcMatch.filter(f => !f.startsWith('__'));
    
    // Detect patterns
    const hasErrorHandling = code.includes('try:') || code.includes('except');
    const hasLogging = code.includes('logging') || code.includes('logger');
    const hasTypeHints = code.includes(': str') || code.includes(': int') || code.includes('-> ');
    const hasDocstrings = code.includes('"""') || code.includes("'''");
    const hasDecorators = (code.match(/@\w+/g) || []).length > 0;
    const hasDataclass = code.includes('@dataclass');
    const hasValidation = code.includes('raise ') || code.includes('ValueError') || code.includes('assert');
    
    // Build description
    let desc = {
      title: '',
      summary: '',
      keyPoints: [],
      features: [],
      structure: ''
    };
    
    // Generate title based on what was requested
    const promptLower = prompt?.toLowerCase() || '';
    if (promptLower.includes('auth')) {
      desc.title = '🔐 Authentication Module';
    } else if (promptLower.includes('api') || promptLower.includes('fetch')) {
      desc.title = '🌐 API Integration';
    } else if (promptLower.includes('database') || promptLower.includes('db')) {
      desc.title = '🗄️ Database Handler';
    } else if (promptLower.includes('rate limit')) {
      desc.title = '⏱️ Rate Limiter';
    } else if (promptLower.includes('cache')) {
      desc.title = '💾 Caching System';
    } else if (promptLower.includes('test')) {
      desc.title = '🧪 Testing Utilities';
    } else if (hasClass && classMatch.length > 0) {
      desc.title = `📦 ${classMatch[0]} Class`;
    } else if (publicMethods.length > 0) {
      desc.title = `⚡ ${publicMethods[0]} Function`;
    } else {
      desc.title = '✨ Generated Code';
    }
    
    // Generate summary
    if (hasClass && classMatch.length > 0) {
      const className = classMatch[0];
      const methodCount = publicMethods.length;
      desc.summary = `I've created a **${className}** class with ${methodCount} method${methodCount !== 1 ? 's' : ''} that implements ${promptLower.includes('auth') ? 'secure authentication logic' : promptLower.includes('api') ? 'robust API communication' : promptLower.includes('pool') ? 'connection pooling' : 'the requested functionality'}. The code follows Python best practices and includes proper error handling.`;
    } else if (publicMethods.length > 0) {
      desc.summary = `I've implemented a **${publicMethods[0]}** function${publicMethods.length > 1 ? ` along with ${publicMethods.length - 1} helper function${publicMethods.length > 2 ? 's' : ''}` : ''} that ${promptLower.replace(/create|build|write|generate|a |an /gi, '').trim() || 'performs the requested operation'}. The implementation is production-ready with comprehensive error handling.`;
    } else {
      desc.summary = `Here's a Python implementation that ${promptLower.replace(/create|build|write|generate|a |an /gi, '').trim() || 'addresses your requirements'}. The code is structured for maintainability and follows best practices.`;
    }
    
    // Key implementation points
    if (hasClass && classMatch.length > 0) {
      desc.keyPoints.push(`**${classMatch[0]}** - Main class encapsulating the logic`);
    }
    if (publicMethods.length > 0) {
      const mainMethods = publicMethods.slice(0, 3);
      mainMethods.forEach(m => {
        desc.keyPoints.push(`\`${m}()\` - ${m.includes('init') ? 'Initializes the instance' : m.includes('get') ? 'Retrieves data' : m.includes('set') ? 'Updates configuration' : m.includes('validate') ? 'Validates input' : m.includes('create') ? 'Creates new resource' : m.includes('delete') ? 'Removes resource' : m.includes('update') ? 'Updates existing data' : 'Core functionality'}`);
      });
    }
    
    // Features detected
    if (hasAsync) desc.features.push('**Async/await** for non-blocking I/O operations');
    if (hasErrorHandling) desc.features.push('**Exception handling** with proper error recovery');
    if (hasTypeHints) desc.features.push('**Type hints** for better IDE support and documentation');
    if (hasDocstrings) desc.features.push('**Docstrings** explaining function behavior');
    if (hasLogging) desc.features.push('**Logging** for debugging and monitoring');
    if (hasDecorators) desc.features.push('**Decorators** for clean, reusable patterns');
    if (hasValidation) desc.features.push('**Input validation** preventing bad data');
    if (hasDataclass) desc.features.push('**Dataclasses** for clean data structures');
    if (code.includes('hashlib') || code.includes('bcrypt')) desc.features.push('**Secure hashing** for sensitive data');
    if (code.includes('retry') || code.includes('backoff')) desc.features.push('**Retry logic** with exponential backoff');
    if (code.includes('threading') || code.includes('asyncio')) desc.features.push('**Concurrency** support built-in');
    
    // Structure info
    desc.structure = `${lines} lines • ${imports} import${imports !== 1 ? 's' : ''}${classMatch.length ? ` • ${classMatch.length} class${classMatch.length > 1 ? 'es' : ''}` : ''}${publicMethods.length ? ` • ${publicMethods.length} function${publicMethods.length > 1 ? 's' : ''}` : ''}`;
    
    return desc;
  };
  
  // Normalize agent name to consistent format
  const normalizeAgentName = (name) => {
    if (!name) return 'Unknown';
    const lower = name.toLowerCase().replace(' agent', '').replace('_', ' ');
    // Capitalize first letter
    return lower.charAt(0).toUpperCase() + lower.slice(1);
  };
  
  // Deduplicate and merge fixes by normalized agent name
  const deduplicateFixes = (fixes) => {
    const merged = {};
    for (const item of fixes) {
      const normalizedName = normalizeAgentName(item.agent);
      if (!merged[normalizedName]) {
        merged[normalizedName] = { agent: normalizedName, fixes: [] };
      }
      // Add fixes if not already present (check by description)
      const existingDescs = new Set(merged[normalizedName].fixes.map(f => 
        typeof f === 'object' ? f.description : f
      ));
      for (const fix of (item.fixes || [])) {
        const desc = typeof fix === 'object' ? fix.description : fix;
        if (!existingDescs.has(desc)) {
          merged[normalizedName].fixes.push(fix);
          existingDescs.add(desc);
        }
      }
    }
    return Object.values(merged).filter(m => m.fixes.length > 0);
  };
  
  // Calculate deduplicated total fixes
  const deduplicatedFixes = result?.all_fixes ? deduplicateFixes(result.all_fixes) : [];
  const totalFixes = deduplicatedFixes.reduce((sum, f) => sum + (f.fixes?.length || 0), 0);
  
  return (
    <div className={`genui-v2 ${isResizing ? 'resizing' : ''}`}>
      {/* Minimal Sidebar */}
      <aside className="genui-v2-sidebar">
        <div className="sidebar-brand">
          <Link to="/" className="brand-link">
            <span className="brand-icon">🚀</span>
            <span className="brand-text">Uber Code</span>
          </Link>
        </div>
        
        <nav className="sidebar-actions">
          <button className="action-btn primary" onClick={createSession}>
            <span>+</span>
            <span>New</span>
          </button>
          <Link to="/dashboard" className="action-btn">
            <span>📊</span>
            <span>History</span>
          </Link>
          <button 
            className={`action-btn ${showSettings ? 'active' : ''}`}
            onClick={() => setShowSettings(!showSettings)}
          >
            <span>⚙️</span>
            <span>Settings</span>
          </button>
        </nav>
        
        {showSettings && (
          <div className="settings-dropdown">
            <label>Groq API Key</label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="gsk_..."
            />
            <button onClick={saveApiKey}>Save</button>
          </div>
        )}
        
        <div className="sidebar-footer">
          <UserMenu />
        </div>
      </aside>
      
      {/* Main Chat Area */}
      <main 
        className={`genui-v2-main ${showRightPanel && (streamingCode || result) ? 'with-panel' : ''}`}
        style={showRightPanel && (streamingCode || result) ? { maxWidth: `calc(100vw - 72px - ${panelWidth}px)` } : {}}
      >
        {/* Header */}
        <header className="main-header">
          <div className="header-title">
            <h1>{sessionTitle}</h1>
            <Badge variant="primary" size="sm">GenUI</Badge>
          </div>
          {(streamingCode || result) && (
            <button 
              className="panel-toggle"
              onClick={() => setShowRightPanel(!showRightPanel)}
            >
              {showRightPanel ? '◀ Hide Panel' : '▶ Show Panel'}
            </button>
          )}
        </header>
        
        {/* Floating Workflow Indicator */}
        {loading && (
          <motion.div 
            className="floating-workflow"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <div className="workflow-steps">
              {['Code Gen', 'Validate', 'Test', 'Secure'].map((step, i) => {
                const agents = ['code_generator', 'validator', 'testing', 'security'];
                const isActive = activeAgent?.toLowerCase().includes(agents[i].split('_')[0]);
                const isDone = agentMessages.some(m => 
                  m.agent?.toLowerCase().includes(agents[i].split('_')[0]) && m.type === 'result'
                );
                return (
                  <div key={step} className={`workflow-step ${isActive ? 'active' : ''} ${isDone ? 'done' : ''}`}>
                    <span className="step-icon">{['⚡', '✓', '🧪', '🛡️'][i]}</span>
                    <span className="step-name">{step}</span>
                  </div>
                );
              })}
            </div>
          </motion.div>
        )}
        
        {/* Messages */}
        <div className="messages-container">
          {messages.length === 0 && !loading && (
            <div className="welcome-screen">
              <div className="welcome-content">
                <div className="welcome-icon">✨</div>
                <h2>Uber Code Generator</h2>
                <p>AI-powered code generation with real-time Generative UI</p>
                
                <div className="quick-prompts">
                  <h4>Quick Start</h4>
                  <div className="prompt-grid">
                    <button onClick={() => setPrompt('Create a Python function to fetch API data with retry logic and error handling')}>
                      <span>🔄</span>
                      <span>API with Retry</span>
                    </button>
                    <button onClick={() => setPrompt('Create a user authentication class with password hashing')}>
                      <span>🔐</span>
                      <span>Auth Class</span>
                    </button>
                    <button onClick={() => setPrompt('Build a rate limiter decorator for API endpoints')}>
                      <span>⏱️</span>
                      <span>Rate Limiter</span>
                    </button>
                    <button onClick={() => setPrompt('Create a database connection pool manager')}>
                      <span>🗄️</span>
                      <span>DB Pool</span>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          <div className="messages-list">
            {messages.map((msg, index) => (
              <motion.div 
                key={index}
                className={`message ${msg.role} ${editingMessageIndex === index ? 'editing' : ''}`}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
              >
                <div className="message-avatar">
                  {msg.role === 'user' ? '👤' : '🤖'}
                </div>
                <div className="message-body">
                  {/* Editable user message */}
                  {msg.role === 'user' && editingMessageIndex === index ? (
                    <form className="edit-message-form" onSubmit={handleSaveEdit}>
                      <textarea
                        value={editingContent}
                        onChange={(e) => setEditingContent(e.target.value)}
                        autoFocus
                        rows={3}
                      />
                      <div className="edit-actions">
                        <button type="button" className="cancel-btn" onClick={handleCancelEdit}>
                          Cancel
                        </button>
                        <button type="submit" className="save-btn" disabled={!editingContent.trim()}>
                          Save & Submit
                        </button>
                      </div>
                    </form>
                  ) : (
                    <>
                      <p>{msg.content}</p>
                      {msg.role === 'user' && !loading && (
                        <button 
                          className="edit-message-btn"
                          onClick={() => handleEditMessage(index, msg.content)}
                          title="Edit message"
                        >
                          ✏️
                        </button>
                      )}
                    </>
                  )}
                  
                  {/* Code description for assistant messages with results */}
                  {msg.role === 'assistant' && msg.hasResult && result && (() => {
                    const desc = generateDescription(result.code, messages[index - 1]?.content);
                    return desc ? (
                      <div className="code-description">
                        <div className="desc-header">
                          <span className="desc-title">{desc.title}</span>
                        </div>
                        <p className="desc-summary" dangerouslySetInnerHTML={{ 
                          __html: desc.summary.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') 
                        }} />
                        
                        {desc.keyPoints.length > 0 && (
                          <div className="desc-key-points">
                            <span className="section-label">Key Components:</span>
                            <ul>
                              {desc.keyPoints.map((point, i) => (
                                <li key={i} dangerouslySetInnerHTML={{ 
                                  __html: point.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/`(.*?)`/g, '<code>$1</code>') 
                                }} />
                              ))}
                            </ul>
                          </div>
                        )}
                        
                        {desc.features.length > 0 && (
                          <div className="desc-features-section">
                            <span className="section-label">Features:</span>
                            <ul className="desc-features">
                              {desc.features.slice(0, 5).map((feat, i) => (
                                <li key={i} dangerouslySetInnerHTML={{ 
                                  __html: feat.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') 
                                }} />
                              ))}
                            </ul>
                          </div>
                        )}
                        
                        <div className="desc-footer">
                          <span className="desc-structure">{desc.structure}</span>
                          {totalFixes > 0 && (
                            <span className="desc-fixes">🔧 {totalFixes} auto-fixes applied</span>
                          )}
                        </div>
                      </div>
                    ) : null;
                  })()}
                </div>
              </motion.div>
            ))}
            
            {loading && (
              <motion.div 
                className="message assistant"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              >
                <div className="message-avatar">🤖</div>
                <div className="message-body">
                  <div className="typing-indicator">
                    <span className="typing-text">
                      {activeAgent ? `${getAgentIcon(activeAgent)} ${activeAgent}` : 'Processing'}
                    </span>
                    <span className="typing-dots">
                      <span>.</span><span>.</span><span>.</span>
                    </span>
                  </div>
                </div>
              </motion.div>
            )}
            
            <div ref={messagesEndRef} />
          </div>
        </div>
        
        {/* Input */}
        <div className="input-container">
          <form onSubmit={handleSubmit} className="input-form">
            <input
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Describe the code you want to generate..."
              disabled={loading}
            />
            <button type="submit" disabled={loading || !prompt.trim()}>
              {loading ? (
                <span className="spinner"></span>
              ) : (
                <span>→</span>
              )}
            </button>
          </form>
          <p className="input-hint">Powered by Llama 3.3 via Groq • 4 AI Agents</p>
        </div>
      </main>
      
      {/* Right Panel - Results */}
      <AnimatePresence>
        {showRightPanel && (streamingCode || result) && (
          <>
            {/* Resize Handle */}
            <div 
              className={`panel-resize-handle ${isResizing ? 'active' : ''}`}
              onMouseDown={startResize}
            >
              <div className="resize-grip">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
            
            <motion.aside 
              className="genui-v2-panel"
              style={{ width: panelWidth }}
              initial={{ width: 0, opacity: 0 }}
              animate={{ width: panelWidth, opacity: 1 }}
              exit={{ width: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
            >
              {/* Panel Tabs */}
              <div className="panel-tabs">
                <button 
                  className={rightPanelTab === 'code' ? 'active' : ''}
                  onClick={() => setRightPanelTab('code')}
                >
                  <span>📝</span> Code
                  {loading && <span className="live-dot"></span>}
                </button>
                <button 
                  className={rightPanelTab === 'agents' ? 'active' : ''}
                  onClick={() => setRightPanelTab('agents')}
                >
                  <span>🤖</span> Agents
                  {agentMessages.length > 0 && <span className="count">{agentMessages.length}</span>}
                </button>
                <button 
                  className={rightPanelTab === 'fixes' ? 'active' : ''}
                  onClick={() => setRightPanelTab('fixes')}
                >
                  <span>🔧</span> Fixes
                  {totalFixes > 0 && <span className="count">{totalFixes}</span>}
                </button>
            </div>
            
            {/* Panel Content */}
            <div className="panel-content">
              {/* Code Tab */}
              {rightPanelTab === 'code' && (
                <div className="code-panel">
                  <div className="code-header">
                    <div className="code-stats">
                      <span>{streamingStats.lines || result?.code?.split('\n').length || 0} lines</span>
                      {result?.code_was_fixed && (
                        <Badge variant="success" size="sm">✓ Fixed</Badge>
                      )}
                    </div>
                    <div className="code-actions">
                      {result?.original_code && (
                        <button onClick={() => setShowOriginal(!showOriginal)}>
                          {showOriginal ? 'Show Fixed' : 'Show Original'}
                        </button>
                      )}
                      <button onClick={copyCode}>
                        {copyStatus || 'Copy'}
                      </button>
                    </div>
                  </div>
                  
                  <div className="code-view">
                    <CodeBlock
                      code={showOriginal ? result?.original_code : (result?.code || streamingCode)}
                      language="python"
                      lineNumbers={true}
                      maxHeight="calc(100vh - 250px)"
                    />
                  </div>
                  
                  {loading && (
                    <div className="streaming-bar">
                      <span className="pulse">●</span>
                      <span>Streaming... {streamingStats.lines} lines</span>
                    </div>
                  )}
                </div>
              )}
              
              {/* Agents Tab */}
              {rightPanelTab === 'agents' && (
                <div className="agents-panel">
                  {/* Stats Row */}
                  {result && (
                    <div className="stats-row">
                      <div className="stat-item">
                        <span className="stat-value">{result.code?.split('\n').length || 0}</span>
                        <span className="stat-label">Lines</span>
                      </div>
                      <div className="stat-item">
                        <span className="stat-value">{totalFixes}</span>
                        <span className="stat-label">Fixes</span>
                      </div>
                      <div className="stat-item">
                        <span className="stat-value">
                          {result.stats?.totalDuration ? `${result.stats.totalDuration}s` : '-'}
                        </span>
                        <span className="stat-label">Time</span>
                      </div>
                    </div>
                  )}
                  
                  {/* Agent Feed - Deduplicated */}
                  <div className="agent-feed">
                    <AnimatePresence>
                      {(() => {
                        // Deduplicate agent messages by normalizing names and keeping latest
                        const seen = new Map();
                        const deduped = [];
                        for (const msg of agentMessages) {
                          const key = `${normalizeAgentName(msg.agent)}-${msg.type}`;
                          if (!seen.has(key) || msg.type === 'result') {
                            seen.set(key, deduped.length);
                            deduped.push({ ...msg, agent: normalizeAgentName(msg.agent) });
                          }
                        }
                        return deduped;
                      })().map((msg, i) => (
                        <motion.div
                          key={i}
                          className={`agent-item ${msg.type}`}
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: i * 0.03 }}
                        >
                          <span className="agent-icon">{getAgentIcon(msg.agent)}</span>
                          <div className="agent-info">
                            <span className="agent-name">{msg.agent}</span>
                            <span className="agent-msg">{msg.message}</span>
                          </div>
                          <span className="agent-time">{msg.time}</span>
                        </motion.div>
                      ))}
                    </AnimatePresence>
                  </div>
                  
                  {/* Agent Results */}
                  {result && (
                    <div className="agent-results">
                      <Tabs
                        tabs={[
                          {
                            id: 'validation',
                            label: 'Validator',
                            icon: '✓',
                            content: (
                              <div className="agent-detail">
                                <p>{result.validation?.message || 'Code validated'}</p>
                                {result.validation?.fixes_applied?.length > 0 && (
                                  <Badge variant="success">{result.validation.fixes_applied.length} fixes</Badge>
                                )}
                              </div>
                            )
                          },
                          {
                            id: 'testing',
                            label: 'Testing',
                            icon: '🧪',
                            content: (
                              <div className="agent-detail">
                                <p>{result.tests?.message || 'Tests passed'}</p>
                                {result.tests?.testability_score && (
                                  <Badge variant="primary">Score: {result.tests.testability_score}</Badge>
                                )}
                              </div>
                            )
                          },
                          {
                            id: 'security',
                            label: 'Security',
                            icon: '🛡️',
                            content: (
                              <div className="agent-detail">
                                <p>{result.security?.message || 'No vulnerabilities'}</p>
                                {result.security?.risk_level && (
                                  <Badge variant={result.security.risk_level === 'low' ? 'success' : 'warning'}>
                                    Risk: {result.security.risk_level}
                                  </Badge>
                                )}
                              </div>
                            )
                          }
                        ]}
                      />
                    </div>
                  )}
                </div>
              )}
              
              {/* Fixes Tab */}
              {rightPanelTab === 'fixes' && (
                <div className="fixes-panel">
                  {result?.all_fixes && result.all_fixes.length > 0 ? (
                    <div className="fixes-list">
                      {deduplicateFixes(result.all_fixes).map((agentFix, i) => (
                        <Expandable
                          key={i}
                          title={`${agentFix.agent} (${agentFix.fixes?.length || 0})`}
                          icon={getAgentIcon(agentFix.agent)}
                          expanded={i === 0}
                        >
                          {agentFix.fixes?.map((fix, j) => (
                            <FixCard
                              key={j}
                              agent={agentFix.agent}
                              description={typeof fix === 'object' ? fix.description : fix}
                              severity={fix.severity || 'medium'}
                              before={fix.before}
                              after={fix.after}
                              line={fix.line}
                              expandable={true}
                            />
                          ))}
                        </Expandable>
                      ))}
                    </div>
                  ) : (
                    <div className="empty-state">
                      <span>✓</span>
                      <p>No fixes needed - code generated cleanly!</p>
                    </div>
                  )}
                  
                  {/* Code Diff */}
                  {result?.code_was_fixed && result?.original_code && (
                    <div className="diff-section">
                      <h4>Full Diff</h4>
                      <CodeDiff
                        before={result.original_code}
                        after={result.code}
                        title=""
                        expanded={false}
                      />
                    </div>
                  )}
                </div>
              )}
            </div>
          </motion.aside>
          </>
        )}
      </AnimatePresence>
    </div>
  );
};

export default GenUIChatPageV2;
