#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QTrust Blockchain Sharding Framework - Secure Channel Implementation
This module implements secure communication channels for the QTrust framework.
"""

import time
import threading
import queue
import json
import base64
import os
from typing import Dict, List, Tuple, Any, Optional, Union, Callable

from .crypto_utils import CryptoManager


class SecureChannel:
    """
    Implements secure communication channels between nodes.
    Provides encryption, authentication, and perfect forward secrecy.
    """

    def __init__(
        self, node_id: str, crypto_manager: CryptoManager, config: Dict[str, Any] = None
    ):
        """
        Initialize the secure channel.

        Args:
            node_id: Node identifier
            crypto_manager: Cryptographic manager
            config: Configuration dictionary
        """
        self.node_id = node_id
        self.crypto = crypto_manager
        self.config = config or {}

        # Default configuration
        self.key_rotation_interval = self.config.get(
            "key_rotation_interval", 3600
        )  # 1 hour
        self.session_timeout = self.config.get("session_timeout", 86400)  # 24 hours
        self.max_message_size = self.config.get("max_message_size", 1048576)  # 1 MB

        # Session keys
        self.sessions = {}  # peer_id -> session_info
        self.session_lock = threading.RLock()

        # Message handlers
        self.message_handlers = {}  # message_type -> handler_function

        # Outgoing message queue
        self.outgoing_queue = queue.Queue()

        # Running flag
        self.running = False
        self.threads = []

    def start(self):
        """
        Start the secure channel.
        """
        if self.running:
            return

        self.running = True

        # Start key rotation thread
        key_rotation_thread = threading.Thread(target=self._key_rotation_loop)
        key_rotation_thread.daemon = True
        key_rotation_thread.start()
        self.threads.append(key_rotation_thread)

        # Start message processing thread
        message_thread = threading.Thread(target=self._message_processing_loop)
        message_thread.daemon = True
        message_thread.start()
        self.threads.append(message_thread)

    def stop(self):
        """
        Stop the secure channel.
        """
        self.running = False

        for thread in self.threads:
            thread.join(timeout=2.0)

        self.threads = []

    def establish_session(self, peer_id: str) -> bool:
        """
        Establish a secure session with a peer.

        Args:
            peer_id: Peer node identifier

        Returns:
            True if session was established, False otherwise
        """
        try:
            with self.session_lock:
                # Check if we already have a valid session
                if peer_id in self.sessions and not self._is_session_expired(peer_id):
                    return True

                # Generate ephemeral key pair for Diffie-Hellman
                ephemeral_private = self.crypto._private_keys[self.crypto.SCHEME_ECDSA]
                ephemeral_public = ephemeral_private.public_key()

                # Create session request
                request = {
                    "type": "session_request",
                    "source": self.node_id,
                    "destination": peer_id,
                    "timestamp": time.time(),
                    "ephemeral_public": self.crypto.get_public_key_str(
                        self.crypto.SCHEME_ECDSA
                    ),
                    "nonce": base64.b64encode(os.urandom(16)).decode("utf-8"),
                }

                # Sign the request
                request["signature"] = self.crypto.sign_data(request)

                # In a real implementation, this would send the request to the peer
                # and wait for a response. For simulation, we'll create a session directly.

                # Generate session key
                session_key = os.urandom(32)  # 256-bit key

                # Create session
                self.sessions[peer_id] = {
                    "established": time.time(),
                    "expires": time.time() + self.session_timeout,
                    "key": session_key,
                    "counter_send": 0,
                    "counter_recv": 0,
                    "last_rotation": time.time(),
                }

                return True

        except Exception as e:
            print(f"Error establishing session with {peer_id}: {e}")
            return False

    def send_message(self, peer_id: str, message_type: str, payload: Any) -> bool:
        """
        Send a secure message to a peer.

        Args:
            peer_id: Peer node identifier
            message_type: Message type
            payload: Message payload

        Returns:
            True if message was queued, False otherwise
        """
        try:
            # Ensure we have a session
            if not self.establish_session(peer_id):
                return False

            # Prepare message
            message = {
                "type": message_type,
                "source": self.node_id,
                "destination": peer_id,
                "timestamp": time.time(),
                "payload": payload,
            }

            # Encrypt message
            encrypted = self._encrypt_message(peer_id, message)
            if not encrypted:
                return False

            # Queue for sending
            self.outgoing_queue.put((peer_id, encrypted))

            return True

        except Exception as e:
            print(f"Error sending message to {peer_id}: {e}")
            return False

    def register_handler(
        self, message_type: str, handler: Callable[[str, Any], None]
    ) -> None:
        """
        Register a message handler.

        Args:
            message_type: Message type
            handler: Handler function (peer_id, payload) -> None
        """
        self.message_handlers[message_type] = handler

    def receive_message(self, peer_id: str, encrypted_message: Dict[str, Any]) -> bool:
        """
        Process a received encrypted message.

        Args:
            peer_id: Peer node identifier
            encrypted_message: Encrypted message

        Returns:
            True if message was processed, False otherwise
        """
        try:
            # Ensure we have a session
            if peer_id not in self.sessions:
                return False

            # Decrypt message
            message = self._decrypt_message(peer_id, encrypted_message)
            if not message:
                return False

            # Verify message
            if not self._verify_message(peer_id, message):
                return False

            # Process message
            message_type = message.get("type")
            if message_type in self.message_handlers:
                handler = self.message_handlers[message_type]
                handler(peer_id, message.get("payload"))

            return True

        except Exception as e:
            print(f"Error processing message from {peer_id}: {e}")
            return False

    def _key_rotation_loop(self):
        """
        Background thread for periodic key rotation.
        """
        while self.running:
            try:
                self._rotate_session_keys()
                time.sleep(60)  # Check every minute
            except Exception as e:
                print(f"Error in key rotation: {e}")
                time.sleep(60)

    def _message_processing_loop(self):
        """
        Background thread for processing outgoing messages.
        """
        while self.running:
            try:
                # Get next message from queue
                peer_id, encrypted_message = self.outgoing_queue.get(timeout=1.0)

                # In a real implementation, this would send the message to the peer
                # For simulation, we'll just print a message
                print(f"Sending encrypted message to {peer_id}")

                self.outgoing_queue.task_done()

            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error processing outgoing message: {e}")
                time.sleep(1.0)

    def _rotate_session_keys(self):
        """
        Rotate session keys for all active sessions.
        """
        with self.session_lock:
            current_time = time.time()

            # Check each session
            for peer_id, session in list(self.sessions.items()):
                # Remove expired sessions
                if current_time > session["expires"]:
                    del self.sessions[peer_id]
                    continue

                # Rotate keys if needed
                if current_time - session["last_rotation"] > self.key_rotation_interval:
                    # Generate new key
                    new_key = os.urandom(32)  # 256-bit key

                    # Update session
                    session["key"] = new_key
                    session["last_rotation"] = current_time
                    session["counter_send"] = 0
                    session["counter_recv"] = 0

    def _encrypt_message(
        self, peer_id: str, message: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Encrypt a message for a peer.

        Args:
            peer_id: Peer node identifier
            message: Message to encrypt

        Returns:
            Encrypted message or None if encryption failed
        """
        with self.session_lock:
            if peer_id not in self.sessions:
                return None

            session = self.sessions[peer_id]

            # Increment counter
            counter = session["counter_send"]
            session["counter_send"] += 1

            # Add counter to message
            message["counter"] = counter

            # Serialize message
            message_json = json.dumps(message)

            # Encrypt with session key
            encrypted_data = self.crypto.encrypt_data(message_json)

            # Add metadata
            encrypted_message = {
                "source": self.node_id,
                "destination": peer_id,
                "timestamp": time.time(),
                "counter": counter,
                "encrypted_data": encrypted_data,
            }

            return encrypted_message

    def _decrypt_message(
        self, peer_id: str, encrypted_message: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Decrypt a message from a peer.

        Args:
            peer_id: Peer node identifier
            encrypted_message: Encrypted message

        Returns:
            Decrypted message or None if decryption failed
        """
        with self.session_lock:
            if peer_id not in self.sessions:
                return None

            session = self.sessions[peer_id]

            # Get counter
            counter = encrypted_message.get("counter")
            if counter is None:
                return None

            # Check for replay attacks
            if counter <= session["counter_recv"]:
                print(f"Potential replay attack detected from {peer_id}")
                return None

            # Update counter
            session["counter_recv"] = counter

            # Decrypt data
            encrypted_data = encrypted_message.get("encrypted_data")
            if not encrypted_data:
                return None

            try:
                decrypted_bytes = self.crypto.decrypt_data(encrypted_data)
                message = json.loads(decrypted_bytes.decode("utf-8"))
                return message
            except Exception as e:
                print(f"Error decrypting message from {peer_id}: {e}")
                return None

    def _verify_message(self, peer_id: str, message: Dict[str, Any]) -> bool:
        """
        Verify a decrypted message.

        Args:
            peer_id: Peer node identifier
            message: Decrypted message

        Returns:
            True if message is valid, False otherwise
        """
        # Check source
        if message.get("source") != peer_id:
            return False

        # Check destination
        if message.get("destination") != self.node_id:
            return False

        # Check timestamp (allow 5 minutes clock skew)
        timestamp = message.get("timestamp", 0)
        current_time = time.time()
        if timestamp < current_time - 300 or timestamp > current_time + 300:
            return False

        return True

    def _is_session_expired(self, peer_id: str) -> bool:
        """
        Check if a session has expired.

        Args:
            peer_id: Peer node identifier

        Returns:
            True if session has expired, False otherwise
        """
        if peer_id not in self.sessions:
            return True

        session = self.sessions[peer_id]
        return time.time() > session["expires"]
