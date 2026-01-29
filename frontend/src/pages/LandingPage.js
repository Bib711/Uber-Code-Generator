import React from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import UserMenu from '../components/UserMenu';
import './LandingPage.css';

const LandingPage = () => {
  const { isAuthenticated } = useAuth();

  const features = [
    {
      icon: '⚡',
      title: 'Code Generator',
      description: 'AI-powered code generation from natural language prompts'
    },
    {
      icon: '✓',
      title: 'Validator Agent',
      description: 'Automatic code validation and style checking'
    },
    {
      icon: '🧪',
      title: 'Testing Agent',
      description: 'Automated test execution and coverage analysis'
    },
    {
      icon: '🛡️',
      title: 'Security Agent',
      description: 'Vulnerability scanning and security recommendations'
    }
  ];

  const workflow = [
    { step: 1, title: 'Enter Prompt', desc: 'Describe what you want to build' },
    { step: 2, title: 'Generate Code', desc: 'AI creates the initial code' },
    { step: 3, title: 'Validate', desc: 'Code quality checks run' },
    { step: 4, title: 'Test', desc: 'Automated tests execute' },
    { step: 5, title: 'Secure', desc: 'Security scan completes' }
  ];

  return (
    <div className="landing-page">
      {/* Header */}
      <header className="header">
        <div className="container header-content">
          <div className="logo">
            <span className="logo-icon">🤖</span>
            <span className="logo-text">Uber Code Generator</span>
          </div>
          <nav className="nav">
            <a href="#features">Features</a>
            <a href="#workflow">Workflow</a>
            <a href="#about">About</a>
            {isAuthenticated ? (
              <>
                <Link to="/chat" className="btn btn-primary">Go to Chat</Link>
                <UserMenu />
              </>
            ) : (
              <Link to="/login" className="btn btn-primary">Sign In</Link>
            )}
          </nav>
        </div>
      </header>

      {/* Hero Section */}
      <section className="hero">
        <div className="container hero-content">
          <div className="hero-badge">🤖 Multi-Agent AI System</div>
          <h1 className="hero-title">
            Generate Production-Ready Code
            <span className="gradient-text"> in Seconds</span>
          </h1>
          <p className="hero-subtitle">
            Leverage the power of 4 specialized AI agents working together to generate, 
            validate, test, and secure your code automatically.
          </p>
          <div className="hero-buttons">
            <Link to="/chat" className="btn btn-large btn-gradient">
              Start Generating
              <span className="btn-arrow">→</span>
            </Link>
            <Link to="/dashboard" className="btn btn-large btn-outline">
              View Dashboard
            </Link>
          </div>
          <div className="hero-stats">
            <div className="stat">
              <span className="stat-value">4</span>
              <span className="stat-label">AI Agents</span>
            </div>
            <div className="stat">
              <span className="stat-value">∞</span>
              <span className="stat-label">Possibilities</span>
            </div>
            <div className="stat">
              <span className="stat-value">0</span>
              <span className="stat-label">Security Issues</span>
            </div>
          </div>
        </div>
        <div className="hero-glow"></div>
      </section>

      {/* Features Section */}
      <section id="features" className="features">
        <div className="container">
          <h2 className="section-title">Powered by 4 Specialized Agents</h2>
          <p className="section-subtitle">Each agent focuses on a specific task in the code generation pipeline</p>
          <div className="features-grid">
            {features.map((feature, index) => (
              <div key={index} className="feature-card" style={{ animationDelay: `${index * 0.1}s` }}>
                <div className="feature-icon">{feature.icon}</div>
                <h3 className="feature-title">{feature.title}</h3>
                <p className="feature-description">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Workflow Section */}
      <section id="workflow" className="workflow">
        <div className="container">
          <h2 className="section-title">How It Works</h2>
          <p className="section-subtitle">A seamless orchestration of AI agents</p>
          <div className="workflow-timeline">
            {workflow.map((item, index) => (
              <div key={index} className="workflow-step">
                <div className="step-number">{item.step}</div>
                <div className="step-content">
                  <h4>{item.title}</h4>
                  <p>{item.desc}</p>
                </div>
                {index < workflow.length - 1 && <div className="step-connector"></div>}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* About Section */}
      <section id="about" className="about">
        <div className="container">
          <h2 className="section-title">About This Project</h2>
          <p className="section-subtitle">A Multi-Agent AI System for Code Generation</p>
          <div className="about-content">
            <div className="about-text">
              <p>
                <strong>Uber Code Generator</strong> is a cutting-edge code generation platform that leverages 
                the power of multiple specialized AI agents working in harmony. Built as a mini project, 
                it demonstrates the potential of multi-agent systems in software development.
              </p>
              <p>
                The system uses <strong>4 specialized agents</strong> - Code Generator, Validator, Testing, and Security - 
                each focusing on a specific aspect of code quality. Together, they ensure that the generated 
                code is not just functional, but also well-structured, tested, and secure.
              </p>
              <div className="tech-stack">
                <h4>Tech Stack</h4>
                <div className="tech-tags">
                  <span className="tech-tag">React</span>
                  <span className="tech-tag">FastAPI</span>
                  <span className="tech-tag">MongoDB</span>
                  <span className="tech-tag">Groq LLM</span>
                  <span className="tech-tag">LangGraph</span>
                  <span className="tech-tag">Google OAuth</span>
                </div>
              </div>
            </div>
            <div className="about-features">
              <div className="about-feature">
                <span className="about-icon">🎯</span>
                <div>
                  <h4>Purpose</h4>
                  <p>Automate code generation with AI-powered quality assurance</p>
                </div>
              </div>
              <div className="about-feature">
                <span className="about-icon">🔄</span>
                <div>
                  <h4>Workflow</h4>
                  <p>Orchestrated multi-agent pipeline for comprehensive code review</p>
                </div>
              </div>
              <div className="about-feature">
                <span className="about-icon">🚀</span>
                <div>
                  <h4>Speed</h4>
                  <p>Ultra-fast generation powered by Groq's LPU inference</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="cta">
        <div className="container cta-content">
          <h2>Ready to Transform Your Coding Experience?</h2>
          <p>Start generating secure, tested, and validated code today.</p>
          <Link to="/chat" className="btn btn-large btn-white">
            Launch Generator
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="footer">
        <div className="container footer-content">
          <div className="footer-brand">
            <span className="logo-icon">🚀</span>
            <span>Uber Code Generator</span>
          </div>
          <p className="footer-text">
            A Multi-Agent Code Generation System | Built with LangGraph & React
          </p>
          <p className="footer-copyright">© 2026 Mini Project</p>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;
