import React, { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Moon, Sun, Menu, Github } from 'lucide-react';
import Home from './pages/Home';
import ApiDocs from './pages/ApiDocs';

function Navigation() {
    const location = useLocation();
    const [theme, setTheme] = useState('dark');

    useEffect(() => {
        document.documentElement.setAttribute('data-theme', theme);
    }, [theme]);

    const toggleTheme = () => {
        setTheme(prev => prev === 'dark' ? 'light' : 'dark');
    };

    return (
        <header className="site-header">
            <div className="header-tabs">
                <Link to="/" className={`header-tab ${location.pathname === '/' ? 'active' : ''}`}>
                    Home
                </Link>
                <Link to="/docs" className={`header-tab ${location.pathname === '/docs' ? 'active' : ''}`}>
                    Documentation
                </Link>
            </div>

            <div className="header-links">
                <a href="https://github.com/shayhacker/LatentMesh" target="_blank" rel="noopener noreferrer" className="github-link">
                    <Github size={16} />
                    GitHub
                </a>
                <button className="theme-toggle" onClick={toggleTheme} aria-label="Toggle theme">
                    {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
                </button>
            </div>

            <button className="mobile-menu-btn" aria-label="Menu">
                <Menu size={20} />
            </button>
        </header>
    );
}

export default function App() {
    return (
        <Router>
            <div className="app-container">
                <Navigation />
                <Routes>
                    <Route path="/" element={<Home />} />
                    <Route path="/docs" element={<ApiDocs />} />
                </Routes>
            </div>
        </Router>
    );
}
