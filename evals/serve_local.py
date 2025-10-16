#!/usr/bin/env python3
"""
Simple local web server to share HTML galleries.
"""

import os
import argparse
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import time

def start_server(port, directory):
    """Start HTTP server"""
    os.chdir(directory)
    
    class CustomHandler(SimpleHTTPRequestHandler):
        def end_headers(self):
            # Add CORS headers for cross-origin requests
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', '*')
            super().end_headers()
    
    server = HTTPServer(('', port), CustomHandler)
    
    print(f"🌐 Server started at http://localhost:{port}")
    print(f"📁 Serving directory: {directory}")
    print(f"📄 HTML files in this directory are now accessible")
    print(f"\n💡 Share with others on the same network:")
    print(f"   http://[YOUR_IP]:{port}")
    print(f"\n🛑 Press Ctrl+C to stop the server")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\n🛑 Server stopped")
        server.shutdown()

def get_local_ip():
    """Get local IP address"""
    import socket
    try:
        # Connect to a remote server to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"

def main():
    parser = argparse.ArgumentParser(description='Start local web server for HTML galleries')
    parser.add_argument('directory', type=str, nargs='?', default='.',
                       help='Directory to serve (default: current directory)')
    parser.add_argument('--port', '-p', type=int, default=8000,
                       help='Port to serve on (default: 8000)')
    parser.add_argument('--open', '-o', action='store_true',
                       help='Open browser automatically')
    parser.add_argument('--ip', action='store_true',
                       help='Show network IP for sharing')
    args = parser.parse_args()
    
    directory = os.path.abspath(args.directory)
    port = args.port
    
    if not os.path.exists(directory):
        print(f"❌ Error: Directory {directory} does not exist!")
        return
    
    # Find HTML files
    html_files = [f for f in os.listdir(directory) if f.endswith('.html')]
    
    if html_files:
        print(f"📄 Found HTML files:")
        for html_file in html_files:
            print(f"   - {html_file}")
        print()
    else:
        print(f"⚠️  No HTML files found in {directory}")
        print()
    
    # Show network IP if requested
    if args.ip:
        local_ip = get_local_ip()
        print(f"🌍 Network IP: {local_ip}")
        print(f"🔗 Share URL: http://{local_ip}:{port}")
        print()
    
    # Open browser if requested
    if args.open and html_files:
        def open_browser():
            time.sleep(1)  # Wait for server to start
            html_file = html_files[0]  # Open first HTML file
            url = f"http://localhost:{port}/{html_file}"
            print(f"🔗 Opening browser: {url}")
            webbrowser.open(url)
        
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
    
    # Start server
    start_server(port, directory)

if __name__ == "__main__":
    main()
