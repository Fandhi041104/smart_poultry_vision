// Utility functions
function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleString('id-ID');
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 25px;
        background: ${type === 'success' ? '#28a745' : type === 'error' ? '#dc3545' : '#667eea'};
        color: white;
        border-radius: 8px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.3);
        z-index: 9999;
        animation: slideIn 0.3s ease;
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Add animation styles
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(400px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(400px); opacity: 0; }
    }
`;
document.head.appendChild(style);

// Check camera status periodically
let cameraCheckInterval;

function startCameraCheck() {
    cameraCheckInterval = setInterval(async () => {
        try {
            const response = await fetch('/api/camera_status');
            const data = await response.json();
            
            const statusEl = document.getElementById('camera-status');
            if (statusEl) {
                if (data.status === 'online') {
                    statusEl.textContent = 'Online';
                    statusEl.style.color = '#28a745';
                } else {
                    statusEl.textContent = 'Offline';
                    statusEl.style.color = '#dc3545';
                }
            }
        } catch (error) {
            console.error('Error checking camera status:', error);
        }
    }, 5000); // Check every 5 seconds
}

// Stop camera check when leaving page
function stopCameraCheck() {
    if (cameraCheckInterval) {
        clearInterval(cameraCheckInterval);
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    // Start camera check if on dashboard
    if (window.location.pathname === '/dashboard') {
        startCameraCheck();
    }
    
    // Mobile menu toggle
    const menuToggle = document.querySelector('.menu-toggle');
    const navMenu = document.querySelector('.nav-menu');
    
    if (menuToggle) {
        menuToggle.addEventListener('click', () => {
            navMenu.classList.toggle('active');
        });
    }
});

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    stopCameraCheck();
});

// Export functions for global use
window.formatTimestamp = formatTimestamp;
window.showNotification = showNotification;