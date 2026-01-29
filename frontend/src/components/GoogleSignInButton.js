import React, { useEffect, useRef } from 'react';
import { useAuth } from '../context/AuthContext';

// Your Google Client ID - loaded from environment or hardcoded for dev
const GOOGLE_CLIENT_ID = process.env.REACT_APP_GOOGLE_CLIENT_ID || '';

const GoogleSignInButton = ({ onSuccess, onError, text = 'signin_with' }) => {
  const buttonRef = useRef(null);
  const { signInWithGoogle } = useAuth();

  useEffect(() => {
    // Load Google Identity Services script
    const script = document.createElement('script');
    script.src = 'https://accounts.google.com/gsi/client';
    script.async = true;
    script.defer = true;
    document.body.appendChild(script);

    script.onload = () => {
      if (window.google && buttonRef.current) {
        window.google.accounts.id.initialize({
          client_id: GOOGLE_CLIENT_ID,
          callback: handleCredentialResponse,
          auto_select: false,
          cancel_on_tap_outside: true,
        });

        window.google.accounts.id.renderButton(buttonRef.current, {
          theme: 'filled_blue',
          size: 'large',
          type: 'standard',
          text: text,  // 'signin_with' or 'signup_with'
          shape: 'rectangular',
          logo_alignment: 'left',
          width: 280,
        });
      }
    };

    return () => {
      // Cleanup
      const scriptElement = document.querySelector('script[src="https://accounts.google.com/gsi/client"]');
      if (scriptElement) {
        scriptElement.remove();
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text]);

  const handleCredentialResponse = async (response) => {
    if (response.credential) {
      const result = await signInWithGoogle(response.credential);
      if (result.success) {
        onSuccess?.(result.user);
      } else {
        onError?.(result.error);
      }
    }
  };

  return (
    <div className="google-signin-container">
      <div ref={buttonRef} id="google-signin-button"></div>
      {!GOOGLE_CLIENT_ID && (
        <p className="google-signin-warning">
          ⚠️ Google Client ID not configured. Add REACT_APP_GOOGLE_CLIENT_ID to your .env file.
        </p>
      )}
    </div>
  );
};

export default GoogleSignInButton;
