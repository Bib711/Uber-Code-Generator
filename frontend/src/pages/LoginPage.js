import React, { useState } from 'react';
import { useNavigate, useLocation, Link, useSearchParams } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { useToast } from '../components/Toast';
import GoogleSignInButton from '../components/GoogleSignInButton';
import './LoginPage.css';

const LoginPage = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [searchParams] = useSearchParams();
  const { isAuthenticated } = useAuth();
  const { showWelcome } = useToast();
  const [error, setError] = useState('');
  
  // Check if user came from signup link
  const isSignup = searchParams.get('mode') === 'signup';

  // Get the page they tried to visit
  const from = location.state?.from?.pathname || '/chat';

  // Redirect if already logged in
  React.useEffect(() => {
    if (isAuthenticated) {
      navigate(from, { replace: true });
    }
  }, [isAuthenticated, navigate, from]);

  const handleSuccess = (user) => {
    console.log('Login successful:', user);
    showWelcome(user.name);
    navigate(from, { replace: true });
  };

  const handleError = (errorMsg) => {
    setError(errorMsg);
  };

  return (
    <div className="login-page">
      <div className="login-container">
        <div className="login-header">
          <Link to="/" className="login-logo">
            <span className="logo-icon">🚀</span>
            <span className="logo-text">Uber Code Generator</span>
          </Link>
        </div>

        <div className="login-card">
          <h1>{isSignup ? 'Create Account' : 'Welcome Back'}</h1>
          <p className="login-subtitle">
            {isSignup 
              ? 'Sign up to start generating code and save your projects'
              : 'Sign in to access the code generator and your chat history'
            }
          </p>

          {error && (
            <div className="login-error">
              <span>⚠️</span> {error}
            </div>
          )}

          <div className="login-google">
            <GoogleSignInButton 
              onSuccess={handleSuccess}
              onError={handleError}
              text={isSignup ? 'signup_with' : 'signin_with'}
            />
          </div>

          <div className="login-divider">
            <span>{isSignup ? 'Quick & secure signup' : 'Secure login with Google'}</span>
          </div>

          <div className="login-features">
            <div className="feature-item">
              <span className="feature-icon">🔒</span>
              <span>Your data is secure</span>
            </div>
            <div className="feature-item">
              <span className="feature-icon">💾</span>
              <span>Save your chat history</span>
            </div>
            <div className="feature-item">
              <span className="feature-icon">⚡</span>
              <span>Access your sessions anywhere</span>
            </div>
          </div>

          <div className="login-switch">
            {isSignup ? (
              <p>Already have an account? <Link to="/login">Sign In</Link></p>
            ) : (
              <p>New here? <Link to="/login?mode=signup">Create Account</Link></p>
            )}
          </div>
        </div>

        <p className="login-footer">
          By signing in, you agree to our Terms of Service and Privacy Policy
        </p>
      </div>
    </div>
  );
};

export default LoginPage;
