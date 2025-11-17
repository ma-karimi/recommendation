#!/usr/bin/env python3
"""
Utility script to check and handle DuckDB lock issues

Usage:
    python check_db_lock.py [--kill-pid PID] [--db-path PATH]
"""
import argparse
import os
import sys
import re
import subprocess
from pathlib import Path

def find_locked_db_processes(db_path: str) -> list:
    """Find processes that might be locking the DuckDB file"""
    locked_processes = []
    
    if not os.path.exists(db_path):
        print(f"Database file does not exist: {db_path}")
        return locked_processes
    
    # Check for lock file (DuckDB creates .lock files)
    lock_file = db_path + ".lock"
    wal_file = db_path + ".wal"
    
    # Try to get process info from lsof (if available)
    try:
        result = subprocess.run(
            ['lsof', db_path],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout:
            for line in result.stdout.split('\n')[1:]:  # Skip header
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 2:
                        pid = parts[1]
                        cmd = ' '.join(parts[8:]) if len(parts) > 8 else 'unknown'
                        locked_processes.append({
                            'pid': pid,
                            'command': cmd,
                            'file': db_path
                        })
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass
    
    # Check for .lock and .wal files
    if os.path.exists(lock_file):
        print(f"⚠️  Lock file found: {lock_file}")
        print("   This indicates a process is using the database")
    
    if os.path.exists(wal_file):
        print(f"⚠️  WAL file found: {wal_file}")
        print("   This indicates a process has an active connection")
    
    return locked_processes

def kill_process(pid: str) -> bool:
    """Kill a process by PID"""
    import signal
    import time
    try:
        pid_int = int(pid)
        # Try graceful termination first
        os.kill(pid_int, signal.SIGTERM)
        print(f"Sent SIGTERM to process {pid}")
        time.sleep(2)
        
        # Check if still running
        try:
            os.kill(pid_int, 0)  # Check if process exists
            print(f"Process {pid} still running, sending SIGKILL...")
            os.kill(pid_int, signal.SIGKILL)
            return True
        except ProcessLookupError:
            print(f"Process {pid} terminated successfully")
            return True
    except (ValueError, ProcessLookupError, PermissionError) as e:
        print(f"Error killing process {pid}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Check and handle DuckDB lock issues')
    parser.add_argument('--db-path', type=str, help='Path to DuckDB file')
    parser.add_argument('--kill-pid', type=str, help='Kill process with this PID')
    parser.add_argument('--list', action='store_true', help='List processes using the database')
    
    args = parser.parse_args()
    
    # Default database path
    if not args.db_path:
        try:
            from settings import load_config
            cfg = load_config()
            args.db_path = os.path.join(cfg.output_dir, "model_data.duckdb")
        except ImportError:
            # Fallback to default path
            args.db_path = "storage/app/recommendation/model_data.duckdb"
            print(f"Using default path: {args.db_path}")
    
    if args.kill_pid:
        print(f"Attempting to kill process {args.kill_pid}...")
        if kill_process(args.kill_pid):
            print("✅ Process killed successfully")
            # Wait a moment for lock to be released
            import time
            time.sleep(1)
            print("Lock should be released now. Try running your script again.")
        else:
            print("❌ Failed to kill process")
            sys.exit(1)
        return
    
    if args.list or not args.kill_pid:
        print(f"Checking for processes using: {args.db_path}")
        print("-" * 60)
        
        processes = find_locked_db_processes(args.db_path)
        
        if processes:
            print(f"\nFound {len(processes)} process(es) using the database:\n")
            for proc in processes:
                print(f"  PID: {proc['pid']}")
                print(f"  Command: {proc['command']}")
                print(f"  File: {proc['file']}")
                print()
            
            print("To kill a process, run:")
            print(f"  python check_db_lock.py --kill-pid <PID>")
            print("\nOr manually:")
            for proc in processes:
                print(f"  kill {proc['pid']}")
        else:
            print("✅ No processes found locking the database")
            print("\nIf you're still getting lock errors:")
            print("  1. Check if the database file exists and is accessible")
            print("  2. Try removing .lock and .wal files (if safe to do so)")
            print("  3. Restart your Python environment")

if __name__ == '__main__':
    import time
    main()

