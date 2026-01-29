import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { useToast } from './Toast';
import './UserMenu.css';

const UserMenu = () => {
  const { user, logout, isAuthenticated } = useAuth();
  const { showSuccess } = useToast();
  const navigate = useNavigate();
  const [showConfirm, setShowConfirm] = useState(false);

  if (!isAuthenticated) {
    return null;
  }

  const handleLogoutClick = () => {
    setShowConfirm(true);
  };

  const handleConfirmLogout = () => {
    logout();
    setShowConfirm(false);
    showSuccess('Successfully logged out. See you soon!');
    navigate('/');
  };

  const handleCancelLogout = () => {
    setShowConfirm(false);
  };

  return (
    <>
      <div className="user-menu">
        <div className="user-info">
          {user.picture ? (
            <img 
              src={user.picture} 
              alt={user.name} 
              className="user-avatar"
              referrerPolicy="no-referrer"
            />
          ) : (
            <div className="user-avatar-placeholder">
              {user.name?.charAt(0)?.toUpperCase() || '?'}
            </div>
          )}
          <span className="user-name">{user.name}</span>
        </div>
        <button onClick={handleLogoutClick} className="logout-btn">
          Logout
        </button>
      </div>

      {/* Logout Confirmation Modal */}
      {showConfirm && (
        <div className="logout-modal-overlay" onClick={handleCancelLogout}>
          <div className="logout-modal" onClick={e => e.stopPropagation()}>
            <div className="logout-modal-icon">👋</div>
            <h3>Leaving so soon?</h3>
            <p>Are you sure you want to logout?</p>
            <div className="logout-modal-actions">
              <button className="btn btn-secondary" onClick={handleCancelLogout}>
                Stay
              </button>
              <button className="btn btn-danger" onClick={handleConfirmLogout}>
                Yes, Logout
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default UserMenu;
